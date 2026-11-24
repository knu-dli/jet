"""Loads the SWE-QA-Pro RL questions, each with a repository, a commit, and a ground truth answer."""

from pathlib import Path

import pyarrow.parquet as pq
from pydantic import BaseModel

PARQUET = Path(__file__).resolve().parents[2] / "third_party/SWE-QA-Pro/train/RL/verl-tool/dataset/train.parquet"


class Question(BaseModel):
    index: int
    question: str
    # "<owner>/<name>"
    repo_name: str
    commit_id: str
    # ground truth 답 (reward_model.ground_truth 와 같다)
    answer: str
    # 저장소를 받아 둘 폴더 이름. 보통 <name>, 이름이 겹치는 저장소만 <owner>__<name>
    repo_dir: str


def load_questions(parquet: Path = PARQUET) -> list[Question]:
    table = pq.read_table(parquet).to_pylist()
    names = []
    for row in table:
        names.append(row["extra_info"]["repo_name"].split("/")[-1])
    questions = []
    for row in table:
        info = row["extra_info"]
        owner, name = info["repo_name"].split("/")
        repo_dir = name
        if names.count(name) > 1:
            repo_dir = f"{owner}__{name}"
        question = Question(
            index=int(info["index"]),
            question=info["question"],
            repo_name=info["repo_name"],
            commit_id=info["commit_id"],
            answer=row["reward_model"]["ground_truth"],
            repo_dir=repo_dir,
        )
        questions.append(question)
    return questions
