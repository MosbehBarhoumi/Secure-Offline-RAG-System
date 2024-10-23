# src/services/chain_service.py
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever

class ChainService:
    """Handle setup and configuration of conversation chains."""
    
    @staticmethod
    def setup_chain(
        hybrid_retriever: ContextualCompressionRetriever,
        llm: Ollama,
        qa_prompt_template: str
    ) -> ConversationalRetrievalChain:
        """Set up the conversational chain."""
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=hybrid_retriever,
            memory=memory,
            return_source_documents=True,
            return_generated_question=True,
            combine_docs_chain_kwargs={
                "prompt": PromptTemplate(
                    template=qa_prompt_template,
                    input_variables=["context", "question"]
                )
            }
        )