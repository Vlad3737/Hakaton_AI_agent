"""Agent components: RAG, memory, model queries."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from chromadb import Client
from chromadb.config import Settings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from src.loaders.pdf_loader import PDFLoader
from src.loaders.tex_loader import TeXLoader
from src.loaders.image_loader import ImageLoader


class RAGSystem:
    """RAG (Retrieval-Augmented Generation) system for papers."""

    def __init__(self, data_dir: Path | str = "data"):
        """Initialize RAG system.

        Args:
            data_dir: Path to paper data directory
        """
        self.data_dir = Path(data_dir)
        self.tex_dir = self.data_dir / "TeX source"
        self.pdf_path = None

        # Find PDF file
        for pdf_file in self.data_dir.glob("*.pdf"):
            self.pdf_path = pdf_file
            break

        # ChromaDB client
        self.chroma_client = Client(
            settings=Settings(
                persist_directory=str(self.data_dir / ".chroma"),
                is_persistent=True,
            )
        )

        # Text collection
        self.text_collection = self.chroma_client.get_or_create_collection(
            name="paper_text",
            metadata={"hnsw:space": "cosine"},
        )

        # Images collection (metadata)
        self.image_collection = self.chroma_client.get_or_create_collection(
            name="paper_images",
            metadata={"hnsw:space": "cosine"},
        )

        # Paper metadata
        self.article_metadata: dict[str, Any] = {}
        self.text_chunks: list[dict] = []
        self.image_metadata: list[dict] = []

    def load_and_chunk(self) -> None:
        """Load and chunk the paper."""
        logger.info("Loading and chunking paper")

        # Load from PDF
        if self.pdf_path and self.pdf_path.exists():
            try:
                pdf_loader = PDFLoader(self.pdf_path)
                pdf_loader.load()
                pdf_text = pdf_loader.extract_text()
                pages = pdf_loader.extract_text_with_pages()
                images = pdf_loader.extract_images()

                logger.info(f"Extracted {len(images)} images from PDF")

                # Save metadata
                self.article_metadata["pdf_pages"] = len(pdf_loader.doc)
                self.article_metadata["pdf_text_length"] = len(pdf_text)

                # Chunk text
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=3000,
                    chunk_overlap=300,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )

                chunks = text_splitter.split_text(pdf_text)

                # Add chunks to database
                for i, chunk in enumerate(chunks):
                    chunk_id = f"pdf_chunk_{i}"
                    self.text_collection.add(
                        documents=[chunk],
                        ids=[chunk_id],
                        metadatas=[{"type": "pdf_chunk", "chunk_idx": i}],
                    )
                    self.text_chunks.append({
                        "id": chunk_id,
                        "text": chunk,
                        "type": "pdf_chunk",
                        "chunk_idx": i,
                    })

                logger.info(f"Added {len(chunks)} chunks from PDF")

            except Exception as e:
                logger.warning(f"Error loading PDF: {e}")

        # Load from LaTeX
        if self.tex_dir.exists():
            try:
                tex_loader = TeXLoader(self.tex_dir)
                tex_loader.discover_files()
                tex_loader.read_main_tex()

                body = tex_loader.body
                figures = tex_loader.extract_figures()
                equations = tex_loader.extract_equations()

                logger.info(f"Extracted {len(figures)} figures and {len(equations)} equations")

                # Save LaTeX metadata
                self.article_metadata["tex_files"] = len(tex_loader.all_tex_files)
                self.article_metadata["figures"] = len(figures)
                self.article_metadata["equations"] = len(equations)

                # Add text chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=3000,
                    chunk_overlap=300,
                    length_function=len,
                    separators=["\n\n", "\n", "\\section", "\\subsection", " ", ""],
                )

                chunks = text_splitter.split_text(body)

                for i, chunk in enumerate(chunks):
                    chunk_id = f"tex_chunk_{i}"
                    self.text_collection.add(
                        documents=[chunk],
                        ids=[chunk_id],
                        metadatas=[{"type": "tex_chunk", "chunk_idx": i}],
                    )
                    self.text_chunks.append({
                        "id": chunk_id,
                        "text": chunk,
                        "type": "tex_chunk",
                        "chunk_idx": i,
                    })

                logger.info(f"Added {len(chunks)} chunks from LaTeX")

                # Add figure metadata
                for fig in figures:
                    img_names = ", ".join(fig["images"]) if fig["images"] else "none"
                    metadata = {
                        "type": "figure",
                        "caption": fig["caption"][:200],
                        "images": img_names,
                        "label": fig["label"],
                    }
                    self.image_collection.add(
                        documents=[fig["caption"]],
                        ids=[f"figure_{fig['label']}"],
                        metadatas=[metadata],
                    )
                    self.image_metadata.append(metadata)

                logger.info(f"Added {len(self.image_metadata)} image metadata")

            except Exception as e:
                logger.warning(f"Error loading LaTeX: {e}")

    def retrieve(self, query: str, n_results: int = 5) -> list[dict]:
        """Search for relevant chunks.

        Args:
            query: Search query
            n_results: Number of results

        Returns:
            List of relevant chunks
        """
        if not self.text_chunks:
            logger.warning("RAG system not initialized")
            return []

        results = self.text_collection.query(
            query_texts=[query],
            n_results=n_results,
        )

        retrieved = []
        for i, doc in enumerate(results["documents"][0]):
            retrieved.append({
                "text": doc,
                "score": results["distances"][0][i] if results.get("distances") else None,
                "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
            })

        return retrieved

    def retrieve_images(self, query: str, n_results: int = 3) -> list[dict]:
        """Search for relevant images.

        Args:
            query: Search query
            n_results: Number of results

        Returns:
            List of relevant images
        """
        if not self.image_metadata:
            return []

        results = self.image_collection.query(
            query_texts=[query],
            n_results=n_results,
        )

        retrieved = []
        for i, doc in enumerate(results["documents"][0]):
            retrieved.append({
                "caption": doc,
                "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
            })

        return retrieved


class GigaChatQuery:
    """Wrapper for GigaChat-2-Max queries."""

    def __init__(self, model: Any | None = None):
        """Initialize querier.

        Args:
            model: GigaChat model (optional, can be set later)
        """
        self.model = model
        self.conversation_history: list[dict] = []

    def set_model(self, model: Any) -> None:
        """Set model.

        Args:
            model: GigaChat model
        """
        self.model = model

    def _build_prompt(self, system_prompt: str, user_prompt: str) -> list:
        """Build message list.

        Args:
            system_prompt: System prompt
            user_prompt: User prompt

        Returns:
            List of messages for model
        """
        messages = [
            SystemMessage(content=system_prompt),
        ]

        # Add conversation history
        for msg in self.conversation_history[-10:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=user_prompt))
        return messages

    def query(
        self,
        prompt: str,
        system_prompt: str = "You are an assistant for analyzing scientific papers. Your answers should be in English.",
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        """Query model.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            temperature: Model temperature
            max_tokens: Maximum tokens

        Returns:
            Model response
        """
        if not self.model:
            raise RuntimeError("Model not set. Call set_model() first.")

        messages = self._build_prompt(system_prompt, prompt)

        try:
            response = self.model.invoke(messages, temperature=temperature)

            # Add to history
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": response.content})

            return response.content

        except Exception as e:
            logger.error(f"Error querying model: {e}")
            return "Error querying model"

    def query_with_rag(
        self,
        question: str,
        rag_system: RAGSystem,
        system_prompt: str = "You are an assistant for analyzing scientific papers. Use the provided context to answer.",
        temperature: float = 0.2,
    ) -> tuple[str, list[dict]]:
        """Query model with RAG context.

        Args:
            question: Question
            rag_system: RAG system
            system_prompt: System prompt
            temperature: Temperature

        Returns:
            Tuple (answer, relevant chunks)
        """
        # Get relevant chunks
        chunks = rag_system.retrieve(question, n_results=5)

        # Form context
        context = "\n\n".join([c["text"] for c in chunks])
        metadata = chunks

        # Form prompt with context
        prompt_template = """
Context from the paper:
{context}

Question:
{question}

Provide a detailed and accurate answer based on the provided context. If the answer is not in the context, say "Answer not found in the paper".

Answer:
"""

        formatted_prompt = prompt_template.format(
            context=context,
            question=question,
        )

        answer = self.query(formatted_prompt, system_prompt, temperature)

        return answer, metadata

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []


class MultimodalQuery:
    """Wrapper for multimodal queries with images."""

    def __init__(self, model: Any | None = None):
        """Initialize multimodal querier.

        Args:
            model: GigaChat model (optional)
        """
        self.model = model
        self.image_loader: ImageLoader | None = None

    def set_model(self, model: Any) -> None:
        """Set model."""
        self.model = model

    def set_image_loader(self, image_loader: ImageLoader) -> None:
        """Set image loader."""
        self.image_loader = image_loader

    def analyze_image(
        self,
        image_filename: str,
        question: str,
        image_dir: Path | str = "data/TeX source",
    ) -> str:
        """Analyze image.

        Args:
            image_filename: Image filename
            question: Question about image
            image_dir: Path to image directory

        Returns:
            Model response
        """
        if not self.model:
            raise RuntimeError("Model not set")

        if not self.image_loader:
            self.image_loader = ImageLoader(image_dir)
            self.image_loader.load_all_images()

        # Get image
        img_data = self.image_loader.get_image_by_filename(image_filename)
        if not img_data:
            return f"Image '{image_filename}' not found"

        # Form prompt
        prompt = f"""
Analyze this image from a scientific paper and answer the question.

Question: {question}

Describe what is shown in the image, interpret the results, and provide a detailed answer.
"""

        # Prepare message with image
        messages = [
            SystemMessage(content="You are an assistant for analyzing scientific papers. Your answers should be in English."),
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_data['base64']}",
                            "detail": "high",
                        },
                    },
                ]
            ),
        ]

        try:
            response = self.model.invoke(messages, temperature=0.2)
            return response.content
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return "Error analyzing image"

    def analyze_with_caption(
        self,
        image_filename: str,
        caption: str,
        question: str,
    ) -> str:
        """Analyze image with caption hint.

        Args:
            image_filename: Image filename
            caption: Caption for image
            question: Question

        Returns:
            Model response
        """
        if not self.model:
            raise RuntimeError("Model not set")

        if not self.image_loader:
            return "Image loader not set"

        img_data = self.image_loader.get_image_by_filename(image_filename)
        if not img_data:
            return f"Image '{image_filename}' not found"

        prompt = f"""
Caption for image: {caption}

Question: {question}

Use the caption and analyze the image to provide a detailed answer.
"""

        messages = [
            SystemMessage(content="You are an assistant for analyzing scientific papers. Your answers should be in English."),
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_data['base64']}"},
                    },
                ]
            ),
        ]

        try:
            response = self.model.invoke(messages, temperature=0.2)
            return response.content
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return "Error analyzing image"
