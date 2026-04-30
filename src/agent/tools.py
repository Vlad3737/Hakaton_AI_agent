"""Agent tools for LangGraph."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from loguru import logger


class AgentTools:
    """Tools for paper analysis agent."""

    def __init__(
        self,
        rag_system: Any,
        query_handler: Any,
        multimodal_handler: Any,
        image_dir: Path | str = "data/TeX source",
    ):
        """Initialize tools.

        Args:
            rag_system: RAG system for search
            query_handler: Model query handler
            multimodal_handler: Multimodal query handler
            image_dir: Path to image directory
        """
        self.rag_system = rag_system
        self.query_handler = query_handler
        self.multimodal_handler = multimodal_handler
        self.image_dir = Path(image_dir)

    def create_tools(self) -> dict[str, Any]:
        """Create tools dict for agent.

        Returns:
            Dict {name: tool}
        """
        return {
            "search_documents": self._search_documents,
            "analyze_image": self._analyze_image,
            "query_model": self._query_model,
            "find_figure": self._find_figure,
            "extract_equation": self._extract_equation,
        }

    def _search_documents(self, query: str, n_results: int = 5) -> str:
        """Search in paper documents.

        Args:
            query: Search query
            n_results: Number of results

        Returns:
            Relevant text fragments
        """
        logger.info(f"Searching documents: {query}")
        chunks = self.rag_system.retrieve(query, n_results=n_results)

        if not chunks:
            return "No information found in the paper for this query."

        result = []
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get("metadata", {})
            chunk_type = metadata.get("type", "chunk")
            result.append(f"[Fragment {i} ({chunk_type})]\n{chunk['text'][:1000]}...")

        return "\n\n".join(result)

    def _analyze_image(self, image_filename: str, question: str) -> str:
        """Analyze image from paper.

        Args:
            image_filename: Image filename (e.g., "Braiding.pdf")
            question: Question about image

        Returns:
            Answer about image
        """
        logger.info(f"Analyzing image: {image_filename}, question: {question}")

        # Check image existence
        image_path = self.image_dir / image_filename
        if not image_path.exists():
            # Try without extension
            for ext in [".pdf", ".png", ".jpg", ".jpeg"]:
                test_path = self.image_dir / f"{image_filename}{ext}"
                if test_path.exists():
                    image_path = test_path
                    break

        if not image_path.exists():
            return f"Image '{image_filename}' not found in folder {self.image_dir}"

        # Try to find RAG metadata
        figure_query = f"figure {image_filename}"
        figures = self.rag_system.retrieve_images(figure_query, n_results=3)

        if figures:
            caption = figures[0].get("caption", "")
            return self.multimodal_handler.analyze_with_caption(
                image_path.name, caption, question
            )

        return self.multimodal_handler.analyze_image(image_path.name, question, str(self.image_dir))

    def _query_model(self, question: str, context: str = "") -> str:
        """Direct query to model.

        Args:
            question: Question
            context: Additional context (optional)

        Returns:
            Model response
        """
        logger.info(f"Direct query to model: {question[:50]}...")

        if context:
            prompt = f"""Context:
{context}

Question:
{question}

Answer:"""
        else:
            prompt = question

        return self.query_handler.query(prompt)

    def _find_figure(self, figure_ref: str) -> str:
        """Find figure by reference in text.

        Args:
            figure_ref: Figure reference (e.g., "Figure 1" or "\ref{fig:test}")

        Returns:
            Figure information
        """
        logger.info(f"Finding figure: {figure_ref}")

        # Search in RAG
        figures = self.rag_system.retrieve_images(figure_ref, n_results=3)

        if not figures:
            return f"Figure '{figure_ref}' not found in the paper."

        result = []
        for fig in figures:
            metadata = fig.get("metadata", {})
            images = metadata.get("images", "none")
            caption = metadata.get("caption", "no caption")
            label = metadata.get("label", "")

            result.append(f"Label: {label}\nImages: {images}\nCaption: {caption}")

        return "\n\n".join(result)

    def _extract_equation(self, equation_ref: str) -> str:
        """Find equation in paper.

        Args:
            equation_ref: Equation reference

        Returns:
            Equation text and context
        """
        logger.info(f"Finding equation: {equation_ref}")

        chunks = self.rag_system.retrieve(equation_ref, n_results=3)

        if not chunks:
            return f"Equation '{equation_ref}' not found in the paper."

        result = []
        for chunk in chunks:
            result.append(chunk["text"][:500])

        return "\n\n".join(result)


def create_search_tool(rag_system: Any):
    """Create search tool for LangGraph."""

    @tool
    def search_documents(query: str, n_results: int = 5) -> str:
        """Search for relevant information in the paper.

        Args:
            query: Search query in Russian or English
            n_results: Number of results (default 5)

        Returns:
            Relevant text fragments from the paper
        """
        chunks = rag_system.retrieve(query, n_results=n_results)
        if not chunks:
            return "No information found in the paper for this query."

        result = []
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get("metadata", {})
            chunk_type = metadata.get("type", "chunk")
            result.append(f"[Fragment {i} ({chunk_type})]\n{chunk['text'][:1000]}...")
        return "\n\n".join(result)

    return search_documents


def create_analyze_image_tool(multimodal_handler: Any, image_dir: Path | str = "data/TeX source"):
    """Create image analysis tool for LangGraph."""

    @tool
    def analyze_image(image_filename: str, question: str) -> str:
        """Analyze image from scientific paper.

        Args:
            image_filename: Image filename (e.g., "Braiding.pdf")
            question: Question about image

        Returns:
            Answer about image content
        """
        image_path = Path(image_dir) / image_filename
        if not image_path.exists():
            return f"Image '{image_filename}' not found"

        return multimodal_handler.analyze_image(image_filename, question, str(image_dir))

    return analyze_image
