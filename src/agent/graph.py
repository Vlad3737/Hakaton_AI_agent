"""Agent graph builder based on LangGraph."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Annotated

from dotenv import load_dotenv
from langchain_gigachat.chat_models import GigaChat
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger

from src.agent.components import RAGSystem, GigaChatQuery, MultimodalQuery
from src.agent.tools import AgentTools, create_search_tool, create_analyze_image_tool


def _reduce_reasoning_steps(
    current: list[str] | None,
    new: list[str] | None,
) -> list[str]:
    """Reducer for reasoning_steps - concatenate lists."""
    if current is None:
        current = []
    if new is None:
        new = []
    return current + new


def _reduce_search_results(
    current: list[str] | None,
    new: list[str] | None,
) -> list[str]:
    """Reducer for search_results - concatenate lists."""
    if current is None:
        current = []
    if new is None:
        new = []
    return current + new


# Agent state definition
class AgentState(MessagesState):
    """State of the agent."""

    question_id: int
    question: str
    answer: str
    search_results: Annotated[list[str], _reduce_search_results]
    images_needed: list[str]
    reasoning_steps: Annotated[list[str], _reduce_reasoning_steps]
    time_taken: float
    completed: bool


def create_agent_graph(model: GigaChat, data_dir: Path = Path("data")) -> Any:
    """Create agent graph.

    Args:
        model: GigaChat model
        data_dir: Path to data directory

    Returns:
        Compiled graph
    """
    data_dir = Path(data_dir)
    tex_dir = data_dir / "TeX source"
    image_dir = tex_dir

    # Initialize components
    logger.info("Initializing RAG system...")
    rag_system = RAGSystem(data_dir)
    rag_system.load_and_chunk()

    logger.info("Initializing query handlers...")
    query_handler = GigaChatQuery(model)
    multimodal_handler = MultimodalQuery(model)

    # Tools
    tools = AgentTools(rag_system, query_handler, multimodal_handler, image_dir)
    search_tool = create_search_tool(rag_system)
    analyze_image_tool = create_analyze_image_tool(multimodal_handler, image_dir)

    # --- Graph nodes ---

    def parse_question(state: AgentState) -> AgentState:
        """Parse question and determine strategy."""
        question_id = state.get("question_id", 1)
        question = state.get("question", "")

        logger.info(f"Parsing question #{question_id}: {question[:50]}...")

        # Check if image is needed - English keywords only
        images_needed = []
        if any(keyword in question.lower() for keyword in ["figure", "picture", "schema", "diagram", "image", "drawing"]):
            # Try to extract image names
            for img_path in image_dir.glob("*.pdf"):
                if img_path.stem.lower() in question.lower():
                    images_needed.append(img_path.name)
            for img_path in image_dir.glob("*.png"):
                if img_path.stem.lower() in question.lower():
                    images_needed.append(img_path.stem + ".png")
            for img_path in image_dir.glob("*.jpg"):
                if img_path.stem.lower() in question.lower():
                    images_needed.append(img_path.stem + ".jpg")

        return {
            "question_id": question_id,
            "question": question,
            "images_needed": images_needed,
            "reasoning_steps": ["Question parsing completed"],
        }

    def search_knowledge(state: AgentState) -> AgentState:
        """Search for knowledge in documents."""
        question = state.get("question", "")
        images_needed = state.get("images_needed", [])

        logger.info(f"Searching knowledge for question...")

        # Search in text
        chunks = rag_system.retrieve(question, n_results=5)
        search_results = [c["text"] for c in chunks]

        # Search images if needed
        if images_needed:
            for img_name in images_needed:
                img_result = rag_system.retrieve_images(img_name, n_results=3)
                search_results.extend([f"Image {img_name}: {f['caption']}" for f in img_result])

        return {
            "search_results": search_results,
            "reasoning_steps": state.get("reasoning_steps", []) + ["Knowledge search completed"],
        }

    def generate_answer(state: AgentState) -> AgentState:
        """Generate answer to the question."""
        question = state.get("question", "")
        search_results = state.get("search_results", [])
        images_needed = state.get("images_needed", [])

        start_time = time.time()
        logger.info(f"Generating answer to question...")

        # Form prompt
        context = "\n\n".join(search_results[:3]) if search_results else "Context not found"

        if images_needed:
            # If image needed, ask for analysis
            prompt = f"""
Question: {question}

Context from the paper:
{context}

Image analysis required: {', '.join(images_needed)}

Please analyze the image and provide a detailed answer based on the image content.
"""
        else:
            prompt = f"""
Question: {question}

Context from the paper:
{context}

Provide a detailed and accurate answer based on the provided context.

Answer:"""

        try:
            answer = model.invoke(prompt, temperature=0.2).content
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            answer = "Error generating answer"

        time_taken = time.time() - start_time

        return {
            "answer": answer,
            "time_taken": time_taken,
            "reasoning_steps": state.get("reasoning_steps", []) + ["Answer generation completed"],
            "completed": True,
        }

    def analyze_image_step(state: AgentState) -> AgentState:
        """Analyze required images."""
        images_needed = state.get("images_needed", [])
        question = state.get("question", "")

        if not images_needed:
            return state

        logger.info(f"Analyzing images: {images_needed}")

        # Analyze each image
        image_analyses = []
        for img_name in images_needed:
            try:
                analysis = multimodal_handler.analyze_image(img_name, question, str(image_dir))
                image_analyses.append(f"{img_name}: {analysis}")
            except Exception as e:
                logger.warning(f"Error analyzing {img_name}: {e}")
                image_analyses.append(f"{img_name}: analysis error")

        return {
            "search_results": state.get("search_results", []) + image_analyses,
            "reasoning_steps": state.get("reasoning_steps", []) + ["Image analysis completed"],
        }

    def format_final_answer(state: AgentState) -> AgentState:
        """Format final answer."""
        question_id = state.get("question_id", 1)
        answer = state.get("answer", "")

        # Check answer length
        if len(answer) > 5000:
            answer = answer[:5000] + "\n\n...[answer truncated]"

        return {
            "answer": f"## Answer {question_id}\n{answer}",
            "completed": True,
        }

    # --- Build graph ---

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("parse_question", parse_question)
    workflow.add_node("analyze_images", analyze_image_step)
    workflow.add_node("search_knowledge", search_knowledge)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("format_answer", format_final_answer)

    # Add edges
    workflow.add_edge("parse_question", "analyze_images")
    workflow.add_edge("parse_question", "search_knowledge")
    workflow.add_edge(["analyze_images", "search_knowledge"], "generate_answer")
    workflow.add_edge("generate_answer", "format_answer")
    workflow.add_edge("format_answer", END)

    # Entry point
    workflow.set_entry_point("parse_question")

    # Compile
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    logger.info("Agent graph built and compiled")

    return app


def run_agent(
    questions: list[str],
    app: Any,
    output_path: Path = Path("output/answers.txt"),
) -> list[str]:
    """Run agent on all questions.

    Args:
        questions: List of questions
        app: Compiled agent graph
        output_path: Path to output file

    Returns:
        List of answers
    """
    answers = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for i, question in enumerate(questions, 1):
        logger.info(f"=== Processing question #{i} ===")
        logger.info(f"Question: {question[:100]}...")

        start_time = time.time()

        # Run agent
        try:
            result = app.invoke(
                {"question_id": i, "question": question},
                config={"configurable": {"thread_id": f"question_{i}"}},
            )

            answer = result.get("answer", "Error generating answer")
            time_taken = result.get("time_taken", 0)

            logger.info(f"Processing time: {time_taken:.2f} seconds")
            logger.info(f"Answer generated, length: {len(answer)} characters")

        except Exception as e:
            logger.error(f"Error processing question {i}: {e}")
            answer = f"## Answer {i}\nError processing question: {str(e)}"
            time_taken = 0

        answers.append(answer)

        # Time limit per question (e.g., 90 seconds)
        if time_taken > 90:
            logger.warning(f"Question {i} took too long: {time_taken:.2f} seconds")

    # Save answers
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(answers))

    logger.info(f"Answers saved to {output_path}")

    return answers


def main(data_dir: Path = Path("data"), output_dir: Path = Path("output")) -> None:
    """Main function to run the agent.

    Args:
        data_dir: Path to data directory
        output_dir: Path to output directory
    """
    load_dotenv()

    # Check environment variables
    credentials = os.getenv("GIGACHAT_CREDENTIALS")
    scope = os.getenv("GIGACHAT_SCOPE")

    if not credentials:
        raise RuntimeError("GIGACHAT_CREDENTIALS is missing from environment variables")
    if not scope:
        raise RuntimeError("GIGACHAT_SCOPE is missing from environment variables")

    # Initialize model
    logger.info("Initializing GigaChat model...")
    model = GigaChat(
        credentials=credentials,
        scope=scope,
        model="GigaChat-2-Max",
        temperature=0.2,
        timeout=120,
        verify_ssl_certs=False,
    )

    # Check model availability
    try:
        test_result = model.invoke("Test")
        logger.info("GigaChat model is available")
    except Exception as e:
        logger.error(f"Error connecting to GigaChat: {e}")
        raise

    # Get questions
    questions_path = data_dir / "questions.txt"
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    with open(questions_path, "r", encoding="utf-8") as f:
        questions_content = f.read()

    # Extract questions (support ## Question N format)
    question_pattern = r"##\s*Question\s+(\d+)\s*(.*?)(?=##\s*Question\s+\d+|$)"
    matches = re.findall(question_pattern, questions_content, re.IGNORECASE | re.DOTALL)

    if matches:
        # Sort by question number
        questions = [q.strip() for _, q in sorted(matches, key=lambda x: int(x[0]))]
    else:
        # Simple format: each non-empty line is a question
        questions = [ln.strip() for ln in questions_content.splitlines() if ln.strip() and not ln.strip().startswith("#")]

    logger.info(f"Found {len(questions)} questions")

    # Create graph
    logger.info("Creating agent graph...")
    app = create_agent_graph(model, data_dir)

    # Run agent
    logger.info("Running agent...")
    answers = run_agent(questions, app, output_dir / "answers.txt")

    logger.info("Agent completed")

    return answers