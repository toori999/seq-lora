import string

PROMPT_WG = (
    "Select one of the choices that answer the following question: {question}\n"
    "Choices: A. {option1}. B. {option2}. Answer:"
)

PROMPT_ARC = (
    "Select one of the choices that answers the following question:\n"
    "{question} Choices: A. {A}. B. {B}. C. {C}. D. {D}. Answer:"
)

PROMPT_OBQA = (
    "Select one of the choices that answers the following question:\n"
    "{question} Choices: A. {A}. B. {B}. C. {C}. D. {D}. Answer:"
)

PROMPT_BOOLQ = (
    "Select one of the choices that answer the following question:\n"
    "Question: {question}\n"
    "Passage: {passage}\n"
    "Choices: A. False. B. True. Answer:"
)

PROMPT_SCIQ = (
    "Select one of the choices that answers the following question:\n"
    "{question} Choices: A. {A}. B. {B}. C. {C}. D. {D}. Answer:"
)

DEFAULT_CHOICE_LABELS = list(string.ascii_uppercase)

MMLU_GROUPS = {
    "science_high": [
        "high_school_physics",
        "high_school_chemistry",
        "high_school_biology",
    ],
    "science_college": [
        "college_physics",
        "college_chemistry",
        "college_biology",
    ],
}

MMLU_EVAL_TASK_PREFIX = "mmlu_"

AGIEVAL_ENGLISH_CONFIGS = [
    "logiqa-en",
    "lsat-ar",
    "lsat-lr",
    "lsat-rc",
    "sat-en",
]

SCIENCEQA_CURRIC_TASK_NAME = "scienceqa_closedchoice_grade2_11"
SCIENCEQA_GRADE12_TASK_NAME = "scienceqa_closedchoice_grade12"
SCIENCEQA_DATASET_NAME = "tcallens/scienceqa-text-only"
SCIENCEQA_GRADE_MIN = 2
SCIENCEQA_GRADE_MAX = 11
SCIENCEQA_TASK_FILTER = "closed choice"

__all__ = [
    "PROMPT_WG",
    "PROMPT_ARC",
    "PROMPT_OBQA",
    "PROMPT_BOOLQ",
    "PROMPT_SCIQ",
    "DEFAULT_CHOICE_LABELS",
    "MMLU_GROUPS",
    "MMLU_EVAL_TASK_PREFIX",
    "AGIEVAL_ENGLISH_CONFIGS",
    "SCIENCEQA_CURRIC_TASK_NAME",
    "SCIENCEQA_GRADE12_TASK_NAME",
    "SCIENCEQA_DATASET_NAME",
    "SCIENCEQA_GRADE_MIN",
    "SCIENCEQA_GRADE_MAX",
    "SCIENCEQA_TASK_FILTER",
]
