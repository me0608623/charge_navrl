import re
import sys

def fix_latex_content(content):
    def process_math_block(match):
        block = match.group(0)
        # 移除 $$ 包裹以便處理內容
        inner = match.group(1)
        
        # 1. 修正截斷的 ext{
        inner = re.sub(r'(?<!\\)ext\{', r'\\text{', inner)
        
        # 2. 修正損壞的 \text{...} } 結構 (例如 \text{好的} }episode)
        inner = re.sub(r'\\text\{([^}]*)\}\s*\}', r'\\text{\1 }', inner)
        
        # 3. 處理中文字符與後續文字的包裹
        # 將 "好的 episode" 這種結構完整包裹
        inner = inner.replace('\\text{好的 }episode', '\\text{好的 episode}')
        inner = inner.replace('\\text{壞的 }episode', '\\text{壞的 episode}')
        inner = inner.replace('\\text{混合 }episode', '\\text{混合 episode}')
        inner = inner.replace('steps', '\\text{steps}')
        
        # 4. 統一底線變數為 \text{...\_...}
        # 具體處理要求的變數
        vars_map = {
            r'value_loss': r'\\text{value\\_loss}',
            r'value_pred': r'\\text{value\\_pred}',
            r'value_target': r'\\text{value\\_target}',
            r'normalized_return': r'\\text{normalized\\_return}',
            r'vf_coeff': r'\\text{vf\\_coeff}',
            r'vf_loss': r'\\text{vf\\_loss}',
            r'target_mean': r'\\text{target\\_mean}',
            r'total_loss': r'\\text{total\\_loss}',
            r'policy_loss': r'\\text{policy\\_loss}',
            r'entropy_loss': r'\\text{entropy\\_loss}',
            r'return': r'\\text{return}'
        }
        
        # 先處理一些複合損壞格式 (Section 6)
        inner = re.sub(r'value_\\text\{target\}', 'value_target', inner)
        inner = re.sub(r'value_\\text\{pred\}', 'value_pred', inner)
        inner = re.sub(r'normalized_\\text\{return\}', 'normalized_return', inner)
        inner = re.sub(r'\\text\{target\}_\{\\text\{mean\}', 'target_mean', inner)
        inner = re.sub(r'\\text\{value\\_pred\}_\{\\text\{mean\}', 'value_pred_mean', inner)
        inner = re.sub(r'value_\\text\{target\}_\{\\text\{mean\}', 'value_target_mean', inner)
        inner = re.sub(r'value_\\text\{target\}_std', 'value_target_std', inner)
        inner = re.sub(r'\\text\{value\}_\{\\text\{init\}_\\text\{bias\}', 'value_init_bias', inner)
        
        # 執行統一替換
        def replace_vars(text):
            # 找到所有包含底線的單字
            def sub_func(m):
                var = m.group(0)
                if var.startswith('\\text{') and var.endswith('}'):
                    return var
                return '\\text{' + var.replace('_', '\\_') + '}'
            
            # 匹配 a_b, a_b_c 等，排除已經在 \text{} 裡的
            # 這裡簡單處理常見模式
            text = re.sub(r'\b[a-zA-Z]+(?:_[a-zA-Z]+)+\b', sub_func, text)
            return text

        inner = replace_vars(inner)
        
        # 修正 Section 4 可能是 }
        inner = inner.replace('\\text{可能是 }', '\\text{可能是 }')
        
        # 修正 if 壓住
        inner = re.sub(r'\bif\b', r'\\text{if}', inner)
        inner = inner.replace('\\text{壓住 }', '\\text{壓住 }')

        # 符號校對
        inner = inner.replace('~', r' \sim ')
        
        return '$$\n' + inner.strip() + '\n$$'

    # 處理所有 $$ 區塊
    content = re.sub(r'\$\$(.*?)\$\$', process_math_block, content, flags=re.DOTALL)
    
    # 處理行內公式或特定標籤 (如摘要中的)
    content = content.replace(r'$\text{vf\_loss}$', r'$\text{vf\_loss}$') # 保持不變
    
    return content

file_path = '/home/aa/Documents/Obsidian Vault/消融前/02_RL基礎_Returns-GAE與NormalizeReturn.md'
with open(file_path, 'r', encoding='utf-8') as f:
    original = f.read()

fixed = fix_latex_content(original)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(fixed)
