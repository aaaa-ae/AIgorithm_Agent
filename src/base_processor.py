import os
import shutil
from abc import ABC, abstractmethod
from typing import Dict, List, Type
import hashlib
import json
from src.prompt import FORMAT


class FileProcessor(ABC):
    """Base class for all file processors"""

    @classmethod
    @abstractmethod
    def get_supported_extensions(cls) -> List[str]:
        """Return list of supported file extensions"""
        pass

    @abstractmethod
    def load(self, file_path: str, force_reprocess: bool = False, **kwargs) -> Dict:
        """
        Main processing method
        Returns dictionary with:
        - content: processed content
        - base_path: cached output folder
        - any possible other information
        """
        pass

    @staticmethod
    def _get_cache_dir(file_path: str, force_reprocess: bool = False):
        """
        Get cache path with hashed filename,
        return cache_path and force_reprocess flag
        if force_reprocess is True, the cache will be deleted
        if content.md not exists, force_reprocess will be set to True

        Args:
            file_path (str): Path to the file
            force_reprocess (bool): Force reprocess the file
        Returns:
            tuple: (cache_path, force_reprocess)
        """
        file_hash = hashlib.md5(os.path.basename(file_path).encode()).hexdigest()[:8]
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        cache_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../output",
            "file_loading",
            f"{base_name}_{file_hash}",
        )
        if force_reprocess:
            shutil.rmtree(cache_path, ignore_errors=True)
        os.makedirs(cache_path, exist_ok=True)

        content_md = os.path.join(cache_path, "content.md")
        if not os.path.exists(content_md):
            force_reprocess = True

        return cache_path, force_reprocess


class ProcessorRegistry:
    """Registry for all available processors"""

    _processors = {}

    @classmethod
    def register(cls, processor: Type[FileProcessor]):
        for ext in processor.get_supported_extensions():
            cls._processors[ext] = processor
        return processor

    @classmethod
    def get_processor(cls, file_path: str) -> FileProcessor:
        ext = os.path.splitext(file_path)[-1].lower()
        processor_cls = cls._processors.get(ext)
        if not processor_cls:
            raise ValueError(f"No processor found for {ext} files")
        return processor_cls()


@ProcessorRegistry.register
class PDFProcessor(FileProcessor):
    @classmethod
    def get_supported_extensions(cls):
        return [".pdf"]

    def load(self, file_path: str, force_reprocess: bool = False, **kwargs):
        """
        Load PDF

        Args:
            file_path (str): Path to the PDF file
            force_reprocess (bool): Force reprocess the file
            method (str): Method to use for processing
                - mineru: Use Mineru to extract text (default)
                - minerloader: Use MinerLoader in Langchain to extract text

        Returns:
            Dict: Processed content
        """
        cache_dir, force_reprocess = self._get_cache_dir(file_path, force_reprocess)

        method = kwargs.get("method", "mineru")
        if method == "mineru":
            return self._load_with_mineru(
                file_path, force_reprocess, cache_dir, **kwargs
            )
        elif method == "minerloader":
            return self._load_with_minerloader(
                file_path, force_reprocess, cache_dir, **kwargs
            )
        else:
            raise ValueError(f"Unknown PDF processing method: {method}")

    def _load_with_mineru(
        self, file_path: str, force_reprocess: bool, cache_dir: str, **kwargs
    ):
        """
        Load PDF with Mineru
        Doc: https://mineru.readthedocs.io/en/latest/user_guide/usage/api.html
        """

        if force_reprocess:
            from magic_pdf.data.data_reader_writer import (
                FileBasedDataWriter,
                FileBasedDataReader,
            )
            from magic_pdf.data.dataset import PymuDocDataset
            from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
            from magic_pdf.config.enums import SupportedPdfParseMethod

            local_image_dir = os.path.join(cache_dir, "images")
            local_md_dir = cache_dir
            image_dir = str(os.path.basename(local_image_dir))
            image_writer, md_writer = FileBasedDataWriter(
                local_image_dir
            ), FileBasedDataWriter(local_md_dir)

            # read pdf bytes
            reader1 = FileBasedDataReader("")
            pdf_bytes = reader1.read(file_path)
            ds = PymuDocDataset(pdf_bytes)

            ## inference
            if ds.classify() == SupportedPdfParseMethod.OCR:
                infer_result = ds.apply(doc_analyze, ocr=True)
                pipe_result = infer_result.pipe_ocr_mode(image_writer)

            else:
                infer_result = ds.apply(doc_analyze, ocr=False, table_enable=False)
                pipe_result = infer_result.pipe_txt_mode(image_writer)

            print(image_dir)  # image
            md_content = pipe_result.get_markdown(image_dir)
            print(type(md_content))  # str
            # 结果写入markdown
            pipe_result.dump_md(md_writer, f"content.md", image_dir)
            print(type(pipe_result))  # <class 'magic_pdf.operators.pipes.PipeResult'>
            # 获取内容列表
            content_list_content = pipe_result.get_content_list(image_dir)
            print(type(content_list_content))  # <list>
            # 转储内容列表
            pipe_result.dump_content_list(md_writer, f"content_list.json", image_dir)
            print(cache_dir)
            return {
                "content": md_content,
                "base_path": cache_dir,
                "content_list": content_list_content,
            }
        else:
            with open(os.path.join(cache_dir, "content.md"), "r") as f:
                md_content = f.read()
            with open(os.path.join(cache_dir, "content_list.json"), "r") as f:
                content_list_content = json.load(f)
            return {
                "content": md_content,
                "base_path": cache_dir,
                "content_list": content_list_content,
            }

    def _load_with_minerloader(
        self, file_path: str, force_reprocess: bool, cache_dir: str, **kwargs
    ):
        if force_reprocess:
            from langchain_community.document_loaders import PDFMinerLoader

            loaded_doc = PDFMinerLoader(file_path).load()
            with open(os.path.join(cache_dir, "content.md"), "w") as f:
                f.write(loaded_doc[0].page_content)
            return {
                "content": loaded_doc[0].page_content,
                "base_path": cache_dir,
            }
        else:
            with open(os.path.join(cache_dir, "content.md"), "r") as f:
                content = f.read()
            return {
                "content": content,
                "base_path": cache_dir,
            }


@ProcessorRegistry.register
class WordProcessor(FileProcessor):
    @classmethod
    def get_supported_extensions(cls):
        return [".docx", ".doc", "ppt", "pptx"]

    def load(self, file_path: str, force_reprocess: bool = False, **kwargs):
        """
        Support MS-DOC, MS-DOCX, MS-PPT, MS-PPTX now


        Args:
            file_path (str): Path to the MS file
            force_reprocess (bool): Force reprocess the file
            method (str): Method to use for processing
                - mineru: Use Mineru to extract text (default)
                - minerloader: Use MinerLoader in Langchain to extract text

        Returns:
            Dict: Processed content
        """
        cache_dir, force_reprocess = self._get_cache_dir(file_path, force_reprocess)

        method = kwargs.get("method", "mineru")
        if method == "mineru":
            return self._load_with_mineru(
                file_path, force_reprocess, cache_dir, **kwargs
            )
        elif method == "minerloader":
            return self._load_with_minerloader(
                file_path, force_reprocess, cache_dir, **kwargs
            )
        else:
            raise ValueError(f"Unknown MS file processing method: {method}")

    def _load_with_mineru(
        self, file_path: str, force_reprocess: bool, cache_dir: str, **kwargs
    ):
        """
        Load MS file with Mineru
        Doc: https://mineru.readthedocs.io/en/latest/user_guide/quick_start/convert_ms_office.html
        """

        if force_reprocess:
            from magic_pdf.data.data_reader_writer import (
                FileBasedDataWriter,
                FileBasedDataReader,
            )
            from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
            from magic_pdf.data.read_api import read_local_office
            from magic_pdf.config.enums import SupportedPdfParseMethod

            # prepare env
            local_image_dir = os.path.join(cache_dir, "images")
            local_md_dir = cache_dir
            image_dir = str(os.path.basename(local_image_dir))
            os.makedirs(local_image_dir, exist_ok=True)
            image_writer, md_writer = FileBasedDataWriter(
                local_image_dir
            ), FileBasedDataWriter(local_md_dir)

            # proc
            ## Create Dataset Instance

            input_file_name = file_path.split(".")[0]
            ds = read_local_office(file_path)[0]

            ## inference
            if ds.classify() == SupportedPdfParseMethod.OCR:
                infer_result = ds.apply(doc_analyze, ocr=True)
                pipe_result = infer_result.pipe_ocr_mode(image_writer)
            else:
                infer_result = ds.apply(doc_analyze, ocr=False, table_enable=False)
                pipe_result = infer_result.pipe_txt_mode(image_writer)

            print(image_dir)  # image
            md_content = pipe_result.get_markdown(image_dir)
            print(type(md_content))  # str
            # 结果写入markdown
            pipe_result.dump_md(md_writer, f"content.md", image_dir)
            print(type(pipe_result))  # <class 'magic_pdf.operators.pipes.PipeResult'>
            # 获取内容列表
            content_list_content = pipe_result.get_content_list(image_dir)
            print(type(content_list_content))  # <list>
            # 转储内容列表
            pipe_result.dump_content_list(md_writer, f"content_list.json", image_dir)
            print(cache_dir)
            return {
                "content": md_content,
                "base_path": cache_dir,
                "content_list": content_list_content,
            }
        else:
            with open(os.path.join(cache_dir, "content.md"), "r") as f:
                md_content = f.read()
            with open(os.path.join(cache_dir, "content_list.json"), "r") as f:
                content_list_content = json.load(f)
            return {
                "content": md_content,
                "base_path": cache_dir,
                "content_list": content_list_content,
            }

    def _load_with_minerloader(
        self, file_path: str, force_reprocess: bool, cache_dir: str, **kwargs
    ):
        pass


@ProcessorRegistry.register
class JSONProcessor(FileProcessor):
    @classmethod
    def get_supported_extensions(cls):
        return [".json"]

    def load(self, file_path: str, force_reprocess: bool = False, **kwargs) -> Dict:
        """
        Load JSON file, download in https://lab.iwhalecloud.com/docchain/docs.

        Args:
            file_path (str): Path to the JSON file

        Returns:
            Dict: {
                "number": Number_of_chunks_and_tables,
                "content": List of text chunks
            }
        """
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        chunks_data = json_data["chunks"]

        format_chunk_lists = []
        for chunk in chunks_data:
            heading_chain = chunk["heading_chain"]
            content = chunk["content"]
            html_content = chunk["html_content"]
            format_chunk = FORMAT["chunk_format_from_json"].format(
                heading_chain=heading_chain, content=content, html_content=html_content
            )
            format_chunk_lists.append(format_chunk)

        return {
            "content": format_chunk_lists,
            "base_path": "",
            "content_list": "",
        }


def load_file(file_path: str, force_reprocess: bool = False, **kwargs) -> Dict:
    processor = ProcessorRegistry.get_processor(file_path)
    return processor.load(file_path, force_reprocess, **kwargs)
