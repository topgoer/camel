"""
这个文件包含了重构后的_find_index_file函数和辅助函数，用于降低认知复杂度
"""
import os
import json
from typing import Dict, List, Any, Optional, Tuple

def _find_index_file(self, index_id: str) -> Optional[str]:
    """查找指定ID的索引文件"""
    self.logger.debug(f"Searching for index file with index_id='{index_id}'")
    
    # 获取搜索目录
    possible_dirs = self._get_search_directories()
    self.logger.debug(f"Will search in directories: {possible_dirs}")
        
    for dir_path in possible_dirs:
        self.logger.debug(f"Checking directory: {dir_path}")
        
        if not os.path.exists(dir_path):
            self.logger.debug(f"Directory '{dir_path}' does not exist")
            continue
            
        # 在目录中搜索匹配的索引文件
        matching_file = self._find_matching_index_in_directory(dir_path, index_id)
        if matching_file:
            return matching_file
                
    return None
    
def _get_search_directories(self) -> List[str]:
    """获取索引文件的搜索目录列表"""
    from app.core.config import settings
    vector_db_dir = settings.VECTOR_STORE_PERSIST_DIR if hasattr(settings, 'VECTOR_STORE_PERSIST_DIR') else os.path.join(self.storage_dir, "storage", "vector_db")
    return [self.indices_dir, vector_db_dir]
    
def _find_matching_index_in_directory(self, dir_path: str, index_id: str) -> Optional[str]:
    """在指定目录中查找匹配的索引文件"""
    for filename in os.listdir(dir_path):
        if not filename.endswith(".json"):
            continue
            
        file_path = os.path.join(dir_path, filename)
        index_data = self._safely_read_json_file(file_path)
        
        if not index_data:
            continue
        
        # 检查索引ID是否匹配
        if self._is_index_match(index_data, filename, index_id):
            return file_path
    
    self.logger.debug(f"No index file with index_id='{index_id}' found in {dir_path}")
    return None
    
def _safely_read_json_file(self, file_path: str) -> Optional[Dict[str, Any]]:
    """安全地读取JSON文件，处理可能的异常"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        self.logger.error(f"Could not decode JSON from file: '{os.path.basename(file_path)}'")
    except Exception as e:
        self.logger.error(f"Error reading file '{os.path.basename(file_path)}': {str(e)}")
    return None
    
def _is_index_match(self, index_data: Dict[str, Any], filename: str, index_id: str) -> bool:
    """检查索引数据是否与目标索引ID匹配"""
    # 检查JSON中的索引ID
    internal_index_id = index_data.get("index_id")
    if internal_index_id == index_id:
        self.logger.debug(f"Match found: File '{filename}' contains index_id='{index_id}'")
        return True
        
    # 备用策略：检查文件名是否包含索引ID
    if index_id in filename:
        self.logger.debug(f"Match found by filename: '{filename}' contains index_id='{index_id}'")
        return True
        
    return False
