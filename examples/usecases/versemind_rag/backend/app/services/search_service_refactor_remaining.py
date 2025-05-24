"""
This file contains refactored implementations of the following functions:
1. _find_index_files_by_collection_or_id (original complexity: 40)
2. _find_embedding_file (original complexity: 49)

The refactored functions have significantly reduced cognitive complexity.
"""
import os
import json
from typing import Dict, List, Any, Optional, Tuple

# REFACTORED _find_index_files_by_collection_or_id FUNCTION
# Original cognitive complexity: 40, target: 15 or less

def _find_index_files_by_collection_or_id_refactored(self, index_id_or_collection: str) -> List[str]:
    """
    查找指定ID或集合名称的索引文件
    
    参数:
        index_id_or_collection: 索引ID或集合名称
        
    返回:
        包含索引文件路径的列表
    """
    self.logger.debug(f"Searching for index files with index_id or collection_name='{index_id_or_collection}'")
    
    # 获取搜索目录
    possible_dirs = self._get_search_directories()
    self.logger.debug(f"Will search in directories: {possible_dirs}")
    
    # 在所有目录中搜索匹配的索引文件
    matching_files = self._find_all_matching_indices(possible_dirs, index_id_or_collection)
    
    # 如果没有找到匹配的文件，尝试单个索引ID回退
    if not matching_files:
        matching_files = self._try_single_index_fallback(index_id_or_collection)
    
    self.logger.debug(f"Found {len(matching_files)} matching index files for '{index_id_or_collection}'")
    return matching_files

def _find_all_matching_indices(self, directories: List[str], index_id_or_collection: str) -> List[str]:
    """在多个目录中查找所有匹配的索引文件"""
    matching_files = []
    
    for dir_path in directories:
        self.logger.debug(f"Checking directory: {dir_path}")
        
        if not os.path.exists(dir_path):
            self.logger.debug(f"Directory '{dir_path}' does not exist")
            continue
            
        matching_files.extend(self._find_matches_in_directory(dir_path, index_id_or_collection))
    
    return matching_files

def _find_matches_in_directory(self, dir_path: str, index_id_or_collection: str) -> List[str]:
    """在单个目录中查找匹配的索引文件"""
    matches = []
    
    for filename in os.listdir(dir_path):
        if not filename.endswith(".json"):
            continue
            
        file_path = os.path.join(dir_path, filename)
        index_data = self._safely_read_json_file(file_path)
        
        if not index_data:
            continue
            
        if self._is_index_match(index_data, filename, index_id_or_collection):
            matches.append(file_path)
            
    return matches

def _is_index_match(self, index_data: Dict, filename: str, index_id_or_collection: str) -> bool:
    """检查索引数据是否与搜索条件匹配"""
    # 检查索引ID
    internal_index_id = index_data.get("index_id")
    if internal_index_id == index_id_or_collection:
        self.logger.debug(f"Match found by index_id: File '{filename}' contains index_id='{index_id_or_collection}'")
        return True
    
    # 检查集合名称
    collection_name = index_data.get("collection_name")
    if collection_name == index_id_or_collection:
        self.logger.debug(f"Match found by collection_name: File '{filename}' belongs to collection='{index_id_or_collection}'")
        return True
        
    # 备用策略：检查文件名是否包含索引ID
    if index_id_or_collection in filename:
        self.logger.debug(f"Match found by filename: '{filename}' contains '{index_id_or_collection}'")
        return True
        
    return False

def _try_single_index_fallback(self, index_id_or_collection: str) -> List[str]:
    """尝试将输入视为单个索引ID（向后兼容）"""
    matching_files = []
    
    self.logger.debug(f"No matching collection found, trying single index fallback for '{index_id_or_collection}'")
    single_index_file = self._find_index_file(index_id_or_collection)
    
    if single_index_file:
        self.logger.debug(f"Found single index file: {single_index_file}")
        matching_files.append(single_index_file)
        
    return matching_files


# REFACTORED _find_embedding_file FUNCTION
# Original cognitive complexity: 49, target: 15 or less

def _find_embedding_file_refactored(self, document_id: str, embedding_id: str = None) -> Optional[str]:
    """查找指定文档的嵌入文件"""
    self.logger.debug(f"Searching for embedding file with document_id='{document_id}' and embedding_id='{embedding_id}'")
    
    # 只在主嵌入目录中查找
    dir_path = self.embeddings_dir
    self.logger.debug(f"Will search in directory: {dir_path}")
    
    if not os.path.exists(dir_path):
        self.logger.debug(f"Embeddings directory '{dir_path}' does not exist")
        return None
        
    # 获取所有可能匹配的文件
    potential_files = self._get_potential_embedding_files(dir_path, document_id)
    
    # 检查每个文件是否符合条件
    for file_path, filename in potential_files:
        embedding_file = self._check_embedding_file_match(file_path, filename, embedding_id)
        if embedding_file:
            return embedding_file
            
    self.logger.debug(f"No matching embedding file found in '{dir_path}'")
    return None

def _get_potential_embedding_files(self, dir_path: str, document_id: str) -> List[Tuple[str, str]]:
    """获取可能包含指定文档ID的所有文件"""
    potential_files = []
    
    for filename in os.listdir(dir_path):
        # 放宽搜索条件，只要包含document_id和.json后缀即可
        if document_id in filename and filename.endswith(".json"):
            self.logger.debug(f"Found potential file: '{filename}'")
            file_path = os.path.join(dir_path, filename)
            potential_files.append((file_path, filename))
            
    return potential_files

def _check_embedding_file_match(self, file_path: str, filename: str, embedding_id: str = None) -> Optional[str]:
    """检查文件是否为所需的嵌入文件"""
    data = self._safely_read_json_file(file_path)
    
    if not data:
        return None
        
    # 打印文件的键，便于调试
    self.logger.debug(f"File '{filename}' contains keys: {list(data.keys())}")
    
    # 如果指定了embedding_id，检查是否匹配
    if not self._check_embedding_id_match(data, filename, embedding_id):
        return None
        
    # 检查文件是否含有嵌入向量相关的键
    if self._has_embedding_keys(data, filename):
        return file_path
        
    # 如果文件名符合嵌入文件命名模式
    if self._has_embedding_filename_pattern(filename):
        return file_path
        
    return None

def _check_embedding_id_match(self, data: Dict, filename: str, embedding_id: str = None) -> bool:
    """检查嵌入ID是否匹配"""
    if not embedding_id:
        return True
        
    internal_embedding_id = data.get("embedding_id")
    if internal_embedding_id and internal_embedding_id != embedding_id:
        self.logger.debug(f"File '{filename}' has embedding_id='{internal_embedding_id}' which doesn't match target '{embedding_id}'")
        return False
        
    return True

def _has_embedding_keys(self, data: Dict, filename: str) -> bool:
    """检查文件是否含有嵌入向量相关的键"""
    for key in ["embeddings", "vectors", "vector"]:
        if key in data:
            self.logger.debug(f"Match found: File '{filename}' contains '{key}' key")
            return True
    return False

def _has_embedding_filename_pattern(self, filename: str) -> bool:
    """检查文件名是否符合嵌入文件命名模式"""
    if "embedded" in filename or "embedding" in filename:
        self.logger.debug(f"Match found by filename pattern: '{filename}'")
        return True
    return False

def _safely_read_json_file(self, file_path: str) -> Optional[Dict]:
    """安全读取JSON文件，处理异常情况"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        self.logger.error(f"Could not decode JSON from file: '{file_path}'")
    except Exception as e:
        self.logger.error(f"Error reading file '{file_path}': {str(e)}")
    
    return None
