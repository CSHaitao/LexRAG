from abc import ABC, abstractmethod
from typing import List, Dict

class PromptBuilder(ABC):
    @abstractmethod
    def build_messages(self, 
                     history: List[Dict], 
                     current_question: str, 
                     articles: List[str]) -> List[Dict]:
        pass

class LegalPromptBuilder(PromptBuilder):
    """Default PromptBuilder"""
    def build_messages(self, history, current_question, articles):
        messages = []
        system_msg = {
            "role": "system",
            "content": "你是一位精通法律知识的专家，致力于为用户提供准确、专业的法律咨询。你的回复应确保严谨、高效，并在风格上与前几轮的回答保持一致（如有）。若用户的问题涉及具体法律条文，应尽可能引用相关法条，以增强回答的权威性。同时，避免提供无关信息，确保回复简明、直接且切中要害。"
        }
        
        if articles:
            system_msg["content"] += "\n\n以下是你可以参考的法条：\n" + "\n".join(
                [f"{i+1}. {art}" for i, art in enumerate(articles)]
            )
        
        messages.append(system_msg)
        for h in history:
            messages.extend([
                {"role": "user", "content": h["user"]},
                {"role": "assistant", "content": h["assistant"]}
            ])
        messages.append({"role": "user", "content": current_question})
        return messages
    
class CustomSystemPromptBuilder(PromptBuilder):
    """Customise system"""
    def __init__(self, system_template: str):
        self.system_template = system_template
        
    def build_messages(self, history, current_question, articles):
        messages = []
        system_content = self.system_template
        if articles:
            system_content += "\n\n以下是你可以参考的法条：\n" + "\n".join(articles)
        
        messages.append({"role": "system", "content": system_content})
        for h in history:
            messages.extend([
                {"role": "user", "content": h["user"]},
                {"role": "assistant", "content": h["assistant"]}
            ])
        messages.append({"role": "user", "content": current_question})
        return messages
    
class FullCustomPromptBuilder(PromptBuilder):
    """Fully customisable PromptBuilder"""
    def __init__(self, build_fn):
        self.build_fn = build_fn  # receive custom constructor
        
    def build_messages(self, history, current_question, articles):
        return self.build_fn(history, current_question, articles)