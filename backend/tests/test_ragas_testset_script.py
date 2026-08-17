import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from app.db.base import Base
from scripts.generate_testset import (
    convert_testset_to_dicts,
    export_testset_to_json,
)
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///:memory:"


@compiles(JSONB, "sqlite")
def compile_jsonb_sqlite(element, compiler, **kw):
    return "JSON"


@pytest.fixture(name="db")
def db_fixture():
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


def test_convert_testset_to_dicts_from_raw_dicts():
    input_data = [
        {
            "user_input": "What is hybrid search?",
            "reference": "Combination of sparse and dense vectors.",
            "reference_contexts": ["Context chunk 1", "Context chunk 2"],
        },
        {
            "question": "What is RAGAS?",
            "ground_truth": "Evaluation framework for RAG systems.",
            "contexts": ["RAGAS evaluates retrieval and generation."],
        },
    ]

    class MockTestset:
        def to_list(self):
            return input_data

    result = convert_testset_to_dicts(MockTestset())
    assert len(result) == 2
    assert result[0]["question"] == "What is hybrid search?"
    assert result[0]["ground_truth"] == "Combination of sparse and dense vectors."
    assert len(result[0]["reference_contexts"]) == 2

    assert result[1]["question"] == "What is RAGAS?"
    assert result[1]["ground_truth"] == "Evaluation framework for RAG systems."
    assert result[1]["reference_contexts"] == [
        "RAGAS evaluates retrieval and generation."
    ]


def test_export_testset_to_json():
    samples = [
        {
            "question": "Sample question?",
            "ground_truth": "Sample ground truth",
            "reference_contexts": ["Ctx 1", "Ctx 2"],
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "testset.json")
        export_testset_to_json(samples, out_path)

        assert os.path.exists(out_path)
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["question"] == "Sample question?"
        assert data[0]["reference_contexts"] == ["Ctx 1", "Ctx 2"]
