import base64
import filetype

def file_to_data_uri(file_path):
    """读取一个文件并将其转换成base64编码利用filetype库,在前面加上mime信息,返回data:uri格式"""
    with open(file_path, 'rb') as file:
        file_content = file.read()
    
    kind = filetype.guess(file_content)
    if kind is None:
        mime_type = 'application/octet-stream'
    else:
        mime_type = kind.mime
    base64_content = base64.b64encode(file_content).decode('utf-8')
    mime_base64 = f"data:{mime_type};base64,{base64_content}"
    return mime_base64


def base64_to_bytes(base64_str: str) -> bytes:
    """
    将Base64编码的字符串（可能带有data:前缀）转换为bytes对象
    参数: base64_str: Base64编码的字符串，可能带有类似"data:image/png;base64,"的前缀
    返回: 解码后的bytes对象
    """
    # 检查是否包含"base64,"前缀，并提取Base64部分
    if "base64," in base64_str:
        # 使用 split 提取 base64, 后面的部分
        try:
            base64_data = base64_str.split("base64,")[1]
        except IndexError:
            raise ValueError("无效的Base64 data URI格式")
    else:
        base64_data = base64_str
    
    # 移除可能的空白字符（如换行符、空格）
    base64_data = base64_data.strip()
    
    try:
        return base64.b64decode(base64_data)
    except base64.binascii.Error as e:
        raise ValueError("无效的Base64编码") from e
