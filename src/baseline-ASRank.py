

#OPENAI-KEY WRITTEN HERE BUT NOT DISPLAYED ON GIT##
env_path = os.path.join(os.path.dirname(__file__), "api.env")
load_dotenv(env_path)
openai.api_key = os.getenv("OPENAI_KEY")


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
    super().__init__(method=method,model_name=model_name,api_key=api_key)
    self.cfg=cfg or ASRankConfig()
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
        for doc in documents:
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

        return documents


# Register into Rankify’s method map
METHOD_MAP["asrank"] = AsRankReranker


#######RUNNING THE CODE###########
from datasets import load_dataset
from rankify.dataset.dataset import Document, Question, Answer, Context

model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# answer_scent function
def answer_scent(question):
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = model.generate(**inputs)
    return tokenizer.decode(output[0], skip_special_tokens=True)


##same will apply with PoliticBias-QA,GenderBias-QA,BBQ and BibleQA##
# Load small IslamQA subset
islamQA_dataset = load_dataset("minhalvp/islamqa", split="train[:10]")

# Prepare documents
documents = []
for ex in islamQA_dataset:
    question = Question(ex["Question"])
    answers = Answer([ex["Full Answer"]])
    contexts = [Context(id=0, text=ex["Full Answer"], score=0)]
    doc = Document(question=question, answers=answers, contexts=contexts)
    documents.append(doc)

# Config
cfg = ASRankConfig(rank_model_name=model_name, device=None)

# Reranker
reranker = AsRankReranker(cfg=cfg, scent_fn=answer_scent)

# Run reranking
rerank_docs = reranker.rank(documents)
##INTEGRATING OUR BIAS METRICS##
##example of evaluation using only one document##
number_of_docs=10
metrics_list_output = []

for idx, (original_doc, perturbed_doc) in enumerate(zip(documents[:number_of_docs], rerank_docs[:number_of_docs]), 1):
    orig_text = [c.text for c in original_doc.contexts]
    pert_text = [c.text for c in perturbed_doc.contexts]

    orig_embeds = [e for e in (get_openai_embedding(t) for t in orig_text) if e is not None]
    pert_embeds = [e for e in (get_openai_embedding(t) for t in pert_text) if e is not None]

    gen_answer = reranker.generate_answer(
        reranker.cfg.baseline_prompt.format(q=original_doc.question.question, scent="bias eval", ctx=pert_text[0] if pert_text else "")
    )

    metrics = {
        "sem_similarity": sem_similarity(orig_embeds, pert_embeds),
        "doc_overlap": doc_overlap(orig_text, pert_text),
        "rep_variance": representation_variance(pert_text, Bias_groups),
        "bias_amp": biasamplicationscore(pert_text, gen_answer)
    }

    metrics_list_output.append(metrics)

    # Print per document
    print(f"\nDocument {idx}")
    for k, v in metrics.items():
        print(f"{k}: {v}")
