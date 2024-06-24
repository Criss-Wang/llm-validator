from llm_benchmark.aspects.cosine_similarity import CosineSimilarity
from llm_benchmark.aspects.json_compare import JsonCompare
from llm_benchmark.aspects.json_structure import JsonStructure
from llm_benchmark.aspects.llm_judge import LLMJudge
from llm_benchmark.aspects.regex_match import RegexMatch
from llm_benchmark.aspects.valid_json import ValidJson
from llm_benchmark.aspects.classification import Classification
from llm_benchmark.aspects.generation import Generation

AVAILABLE_ASPECTS = [
    ValidJson,
    JsonStructure,
    CosineSimilarity,
    RegexMatch,
    JsonCompare,
    LLMJudge,
    Classification,
    Generation,
]

ASPECT_REGISTRY = {_.get_id(): _ for _ in AVAILABLE_ASPECTS}


def create_aspect(aspect_id: str, params: dict = None):
    if params is None:
        params = {}

    aspect_class = ASPECT_REGISTRY[aspect_id]

    return aspect_class(**params)
