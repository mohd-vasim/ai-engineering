"""ChatGPT JSON to DataFrame Converter.

This module provides a class to parse ChatGPT conversation exports
and convert them to a pandas DataFrame for further processing.
"""

import json
from pathlib import Path
from typing import Union, List, Dict, Any

import pandas as pd


class ChatGPTConverter:
    """Convert ChatGPT exported JSON to pandas DataFrame.
    
    The ChatGPT export format is a tree structure where each message has:
    - parent: ID of the parent message
    - children: IDs of child messages
    - message: contains author.role, content.parts, etc.
    
    This class traverses the tree to extract user-assistant conversation pairs.
    """

    def __init__(self):
        """Initialize the converter."""
        self.conversations: List[Dict[str, Any]] = []
        self.df: pd.DataFrame = None

    def load_json(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load ChatGPT JSON export file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Parsed JSON data
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def _get_message_content(self, node: Dict[str, Any]) -> str:
        """Extract text content from a message node.
        
        Args:
            node: Message node from the mapping
            
        Returns:
            Extracted text content
        """
        if not node.get('message'):
            return ""
        
        content = node['message'].get('content', {})
        parts = content.get('parts', [])
        
        if parts and isinstance(parts, list):
            return ' '.join(str(part) for part in parts)
        return ""

    def _get_message_role(self, node: Dict[str, Any]) -> str:
        """Extract author role from a message node.
        
        Args:
            node: Message node from the mapping
            
        Returns:
            Role string ('user', 'assistant', 'system')
        """
        if not node.get('message'):
            return "unknown"
        return node['message'].get('author', {}).get('role', 'unknown')

    def _find_root_node(self, mapping: Dict[str, Dict]) -> str:
        """Find the root node of a conversation.
        
        The root has parent set to 'client-created-root'.
        
        Args:
            mapping: Conversation mapping dictionary
            
        Returns:
            Root node ID
        """
        for node_id, node in mapping.items():
            if node.get('parent') == 'client-created-root':
                return node_id
        return None

    def _traverse_conversation(self, mapping: Dict[str, Dict], conversation_id: str = "unknown") -> List[Dict[str, str]]:
        """Traverse the conversation tree and extract Q&A pairs.
        
        Args:
            mapping: Conversation mapping dictionary
            conversation_id: The conversation ID for grouping
            
        Returns:
            List of conversation turns as dicts with session_id
        """
        turns = []
        session_id = 0
        
        # Find all user messages and find their corresponding assistant responses
        for node_id, node in mapping.items():
            role = self._get_message_role(node)
            
            if role == 'user':
                user_content = self._get_message_content(node)
                
                if not user_content.strip():
                    continue
                
                # Find the assistant response (first child that is an assistant)
                children = node.get('children', [])
                
                for child_id in children:
                    if child_id not in mapping:
                        continue
                    
                    child_node = mapping[child_id]
                    child_role = self._get_message_role(child_node)
                    
                    if child_role == 'assistant':
                        assistant_content = self._get_message_content(child_node)
                        
                        if assistant_content.strip():
                            turns.append({
                                'instruction': user_content,
                                'output': assistant_content,
                                'conversation_id': conversation_id,
                                'session_id': session_id
                            })
                            session_id += 1
                            break  # Only take first assistant response
        
        return turns

    def _extract_conversation_pairs(self, conversation: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract all user-assistant pairs from a single conversation.
        
        Args:
            conversation: Single conversation object from ChatGPT export
            
        Returns:
            List of instruction-output pairs with conversation_id and session_id
        """
        mapping = conversation.get('mapping', {})
        conversation_id = conversation.get('id', 'unknown')
        
        # Find root and traverse from there
        root_id = self._find_root_node(mapping)
        
        if not root_id:
            # Fallback: just collect all user->assistant pairs
            return self._traverse_conversation(mapping)
        
        # Start from root and follow the chain
        turns = []
        current_id = root_id
        visited = set()
        session_id = 0  # Track session within this conversation
        
        while current_id and current_id not in visited:
            visited.add(current_id)
            node = mapping.get(current_id)
            
            if not node:
                break
            
            role = self._get_message_role(node)
            
            # If this is a user message, look for assistant response
            if role == 'user':
                user_content = self._get_message_content(node)
                
                if user_content.strip():
                    # Find assistant response(s) in children
                    children = node.get('children', [])
                    
                    for child_id in children:
                        if child_id not in mapping:
                            continue
                        
                        child_node = mapping[child_id]
                        child_role = self._get_message_role(child_node)
                        
                        if child_role == 'assistant':
                            assistant_content = self._get_message_content(child_node)
                            
                            if assistant_content.strip():
                                turns.append({
                                    'instruction': user_content,
                                    'output': assistant_content,
                                    'conversation_id': conversation_id,
                                    'session_id': session_id
                                })
                                session_id += 1
                                break
            
            # Move to first child
            children = node.get('children', [])
            if children:
                current_id = children[0]
            else:
                break
        
        return turns

    def convert(self, json_data: Union[str, Path, List, Dict]) -> pd.DataFrame:
        """Convert ChatGPT JSON to pandas DataFrame.
        
        Args:
            json_data: Path to JSON file, or loaded JSON data (list/dict)
            
        Returns:
            DataFrame with 'instruction' and 'output' columns
        """
        # Load JSON if path is provided
        if isinstance(json_data, (str, Path)):
            json_data = self.load_json(json_data)
        
        # Ensure it's a list
        if isinstance(json_data, dict):
            json_data = [json_data]
        
        all_pairs = []
        
        for conversation in json_data:
            pairs = self._extract_conversation_pairs(conversation)
            all_pairs.extend(pairs)
        
        self.df = pd.DataFrame(all_pairs)
        return self.df

    def to_csv(self, output_path: Union[str, Path], **kwargs) -> None:
        """Save DataFrame to CSV.
        
        Args:
            output_path: Path to output CSV file
            **kwargs: Additional arguments for to_csv
        """
        if self.df is None:
            raise ValueError("No data to save. Call convert() first.")
        
        self.df.to_csv(output_path, index=False, **kwargs)

    def to_json(self, output_path: Union[str, Path], **kwargs) -> None:
        """Save DataFrame to JSON (Alpaca format).
        
        Args:
            output_path: Path to output JSON file
            **kwargs: Additional arguments for to_json
        """
        if self.df is None:
            raise ValueError("No data to save. Call convert() first.")
        
        self.df.to_json(output_path, orient='records', indent=2, **kwargs)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the converted data.
        
        Returns:
            Dictionary with statistics
        """
        if self.df is None:
            return {"error": "No data loaded"}
        
        return {
            "total_pairs": len(self.df),
            "avg_instruction_length": self.df['instruction'].str.len().mean(),
            "avg_output_length": self.df['output'].str.len().mean(),
            "max_instruction_length": self.df['instruction'].str.len().max(),
            "max_output_length": self.df['output'].str.len().max(),
        }