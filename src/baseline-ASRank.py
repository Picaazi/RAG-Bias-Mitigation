#OPENAI-KEY WRITTEN HERE BUT NOT DISPLAYED ON GIT##

@dataclass
##this class holds all the hyperparameter to control behaviour of AsRank##
class ASRankConfig:
  rank_model_name: str ="t5-base"
  max_answer_tokens: int = 64
  scent_max_tokens:int =128
  device:str=None
  alpha_retriever_prior: float =1.0
  beta_answer_baseline: float= 1.0
  normalize_prior:bool=True
  baseline_prompt:str=("You are a precise QA system.\n"
        "Question: {q}\n"
        "Answer Scent: {scent}\n"
        "Context: {ctx}\n"
        "Generate a concise answer:"
        )
##implementation of AsRank##
class AsRankReranker(BaseRanking):
  def __init__(self,method="asrank",model_name=None,api_key=None,scent_fn=None,cfg=None,**kwargs):
    env_path = os.path.join(os.path.dirname(__file__), "api.env")
    load_dotenv(env_path)
    openai.api_key = os.environ.get("OPENAI_KEY")
    #from google.colab import userdata
    self.api_key=openai.api_key #userdata.get('OPENAI_KEY')
    self.cfg=cfg or ASRankConfig()

    super().__init__(method=method,model_name=cfg.rank_model_name,api_key=api_key)

    self.scent_fn=scent_fn or (lambda q: f"Likely concise answer to: {q}")
    self.device=self.cfg.device or "cuda" if torch.cuda.is_available() else "cpu"
    self.tokenizer=AutoTokenizer.from_pretrained(self.cfg.rank_model_name)
    self.model=AutoModelForSeq2SeqLM.from_pretrained(self.cfg.rank_model_name).to(self.device)
    self.model.eval()

  @torch.no_grad()
  def generate_answer(self,input):
    tokens=self.tokenizer(input,return_tensors="pt",truncation=True,padding=False).to(self.device)
    output=self.model.generate(**tokens,max_new_tokens=self.cfg.max_answer_tokens)
    return self.tokenizer.decode(output[0],skip_special_tokens=True)

  @torch.no_grad()
  ##nnl stands for negative log-likelihood##
  def sequence_nnl(self,input,target):
    enc=self.tokenizer(input,return_tensors="pt",truncation=True,padding=False).to(self.device)
    with self.tokenizer.as_target_tokenizer():
      lab=self.tokenizer(target,return_tensors="pt",truncation=True,padding=False).to(self.device)
    output= self.model(**enc,labels=lab["input_ids"])
    token_count=(lab["input_ids"]!=self.tokenizer.pad_token_id).sum().item()
    return float(output.loss.item()*max(token_count,1))

  def _blend_prior(self, ctx_score: Optional[float], ctx_scores: List[Optional[float]]) -> float:
        if ctx_score is None:
            return 0.0
        if not self.cfg.normalize_prior:
            return float(ctx_score)
        vals = [s for s in ctx_scores if s is not None]
        if len(vals) == 0:
            return float(ctx_score)
        lo, hi = min(vals), max(vals)
        if hi <= lo:
            norm = 0.5
        else:
            norm = (ctx_score - lo) / (hi - lo)
        eps = 1e-6
        norm = min(max(norm, eps), 1 - eps)
        return float(torch.log(torch.tensor(norm / (1 - norm))).item())

  def rank(self, documents: List[Document]):
        print(f"Number of documnets: {len(documents)}")
        idx =1
        for doc in documents:
            print(idx)
            q = doc.question.question
            scent = self.scent_fn(q)

            baseline_input = self.cfg.baseline_prompt.format(q=q, scent=scent,ctx="")
            baseline_answer = self.generate_answer(baseline_input)
            baseline_nll = self.sequence_nnl(baseline_input, baseline_answer)

            prior_scores = [getattr(c, "score", None) for c in doc.contexts]
            scored = []
            for ctx in doc.contexts:
                ctx_text = getattr(ctx, "text", "")
                inp = self.cfg.baseline_prompt.format(q=q, scent=scent, ctx=ctx_text)

                a_i = self.generate_answer(inp)
                nll_doc = self.sequence_nnl(inp, a_i)

                prior = self._blend_prior(getattr(ctx, "score", None), prior_scores)
                s = (-nll_doc) + self.cfg.alpha_retriever_prior * prior - self.cfg.beta_answer_baseline * baseline_nll

                ctx.score = float(s)
                scored.append(ctx)

            doc.reorder_contexts = sorted(scored, key=lambda c: c.score, reverse=True)
            idx +=1
        return documents

  def generate_samples(self,prompt:str,model_name:str,num_samples:int=5,seed:int=42):
      request_lock=threading.Lock()
      last_request_time=0
      MIN_REQUEST_INTERVAL=0.1
      client = OpenAI(api_key=self.api_key)
      samples=[]

      def make_single_request(sample_seed:int):
        nonlocal last_request_time
        with request_lock:
          elapsed=time.time()-last_request_time
          if elapsed<MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL-elapsed)
          last_request_time=time.time()

        try:
          response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=150,
                top_p=0.9,
                seed=sample_seed,
            )

          return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Request failed: {e}")
            return None


      with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        seeds = [seed + i for i in range(num_samples)]
        futures = [executor.submit(make_single_request, s) for s in seeds]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result and len(result.split()) > 10:
                samples.append(result)

      return samples

  def answer_scent(self,question):
    try:
        prompt = f"Generate a brief, insightful answer scent to the following question: {question}"
        samples=self.generate_samples(prompt=prompt,model_name="gpt-3.5-turbo",num_samples=1)
        api_key=self.api_key
          if not api_key:
              raise ValueError("OPENAI_API_KEY environment variable not set")

        # client = OpenAI(api_key=self.api_key)
        # response = client.chat.completions.create(
        #     model="gpt-3.5-turbo",
        #     messages=[{"role": "user", "content": prompt}],
        #     max_tokens=128,
        #     temperature=0.7
        # )


        if samples:
          scent_text=samples[0]
          print("prompt is",prompt)
          print("scent_text is:",scent_text)
          return scent_text or "No scent generated"
        else:
          return f"Likely concise answer to: {question}"


    except AuthenticationError:
        print("OpenAI authentication failed. Using fallback scent.")
        return f"Likely concise answer to: {question}"

    except Exception as e:
        print(f"Error generating scent: {e}. Using fallback scent.")
        return f"Likely concise answer to: {question}"

# Register into Rankify’s method map
METHOD_MAP["asrank"] = AsRankReranker

#######RUNNING THE CODE###########
if __name__ == "__main__":
  model_name = "t5-small"
#   tokenizer = AutoTokenizer.from_pretrained(model_name)
#   model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


  ##loading datasets+corpus combined##

  ##loading MS MARCO from the corpus_loader.py file##
  msmarco_test = load_dataset("ms_marco", "v2.1", split="test[:5]")
  marco_df = corpus_loader.flatten_and_deduplicate(msmarco_test)
  msmarco_corpus = marco_df["passage_text"].dropna().tolist()

  # Loading webis argument framing-19
  webis_url = "https://raw.githubusercontent.com/Picaazi/RAG-Bias-Mitigation/refs/heads/RAG_system/corpus_data/webis-argument-framing.csv"
  webis_test = load_dataset("csv", data_files=webis_url)


  if "conclusion" in webis_test["train"].column_names:
      webis_corpus = [item for item in webis_test["train"]["conclusion"] if item is not None]
  else:
      print("Column 'conclusion' not found in webis dataset. Available columns:", webis_test["train"].column_names)
      webis_corpus = []

  # Loading polNLI
  polNLi_url = "https://raw.githubusercontent.com/Picaazi/RAG-Bias-Mitigation/refs/heads/RAG_system/corpus_data/pol_nli_test.csv"
  polNLi_test = load_dataset("csv", data_files=polNLi_url)


  if "premise" in polNLi_test["train"].column_names:
      polNLi_corpus = [item for item in polNLi_test["train"]["premise"] if item is not None]
  else:
      print("Column 'premise' not found in polNLI dataset. Available columns:", polNLi_test["train"].column_names)
      polNLi_corpus = []

  # Loading SBIC
  SBIC_url = "https://raw.githubusercontent.com/Picaazi/RAG-Bias-Mitigation/refs/heads/main/SBIC.csv"
  SBIC_test = load_dataset("csv", data_files=SBIC_url)


  if "post" in SBIC_test["train"].column_names:
      SBIC_corpus = [item for item in SBIC_test["train"]["post"] if item is not None]
  else:
      print("Column 'post' not found in SBIC dataset. Available columns:", SBIC_test["train"].column_names)
      SBIC_corpus = []

  # Loading GenderBiasQA
  genderbiasQA = multi_dataset_loader.gender_bias()
  genderbiasQA_corpus = genderbiasQA.corpus()

# Combine all corpora

  combined_corpus = list(set(msmarco_corpus + genderbiasQA_corpus + webis_corpus + polNLi_corpus + SBIC_corpus))
  random.shuffle(combined_corpus)  # Randomize for a fair dataset-corpus combo

# Get queries from GenderBiasQA
  queries = genderbiasQA.query()
  print("Loaded dataset,corpus and queries")


# Prepare relevant documents for each query in genderbiasQA##
  documents = []
  test_queries=queries[:10]
  test_docs=combined_corpus[:30]
  metrics_list_output = []

  for q_idx, query_text in enumerate(test_queries):
      contexts = [Context(id=i,text=text) for i,text in enumerate(test_docs)]
      doc = Document(question=Question(query_text),answers=None,contexts=contexts)
      documents.append(doc)
  # Config
  cfg = ASRankConfig(rank_model_name=model_name, device=None)
  reranker = AsRankReranker(cfg=cfg)
  rerank_docs = reranker.rank(documents)
  reranker.scent_fn = reranker.answer_scent
  print("All document reranked")
  word_bag=bias_grps.get_bias_grps()


  ##INTEGRATING OUR BIAS METRICS##
  print("Start running metrics calculation")

  for idx, (original_doc, perturbed_doc) in enumerate(zip(documents, rerank_docs), 1):
    print(f"Current idx: {idx}")
    orig_text = [c.text for c in original_doc.contexts]
    pert_text = [c.text for c in perturbed_doc.contexts]

    print("calling scent_fn")
    scent = reranker.scent_fn(original_doc.question.question)


    print("calling get openai embeddings")
    orig_embeds = [e for e in (client.get_openai_embedding(t) for t in orig_text) if e is not None]
    pert_embeds = [e for e in (client.get_openai_embedding(t) for t in pert_text) if e is not None]

    print("calling reranker generate ans")
    gen_answer = reranker.generate_answer(
        reranker.cfg.baseline_prompt.format(
            q=original_doc.question.question,
            scent="bias eval",
            ctx=pert_text[0] if pert_text else ""
        )
    )

    ##output format per query##
    metrics = {
        "sem_similarity": metrics.sem_similarity(orig_embeds,pert_embeds),
        "doc_overlap": metrics.doc_overlap(orig_text,pert_text),
        "rep_variance": metrics.representation_variance(documents=pert_text,group_set=word_bag),
        "bias_amp": metrics.biasamplicationscore(pert_text,gen_answer)
      }

    metrics_list_output.append({
        "query": original_doc.question.question,
        "answer_scent": scent,
        "top_contexts": pert_text[:10],  # top 10 contexts
        "metrics": metrics
      })




  print("metrics list built per query")


  flat_data = []
  for item in metrics_list_output:
      row = {
          "query": item["query"],
          "answer_scent": item["answer_scent"],
          "top_contexts": " | ".join(item["top_contexts"]) if isinstance(item["top_contexts"], list) else item["top_contexts"],
      }
      for k, v in item["metrics"].items():
          row[k] = v
      flat_data.append(row)

  df = pd.DataFrame(flat_data)
  df.to_csv("output.csv", index=False)
  print("CSV saved as output.csv")

