import os
import re

def convert_to_latex(text):
    # 1. Convert text code blocks that look like math to $$ block math $$
    def replace_code_block(match):
        code_type = match.group(1)
        content = match.group(2).strip()
        # If it's a text block and contains math-like symbols
        if code_type == "text" and any(op in content for op in ["=", "+", "-", "*", "/", "^", "≈", ">", "<"]):
            # Simple conversion logic
            content = content.replace(" * ", " \\cdot ")
            content = content.replace(" × ", " \\times ")
            content = content.replace(" ≈ ", " \\approx ")
            content = content.replace("mean(", "\\operatorname{mean}(")
            content = content.replace("std(", "\\operatorname{std}(")
            content = content.replace("Var(", "\\operatorname{Var}(")
            content = content.replace("MSE(", "\\operatorname{MSE}(")
            content = content.replace("ReLU(", "\\operatorname{ReLU}(")
            content = content.replace("V(s)", "V(s)")
            content = content.replace("target", "\\text{target}")
            content = content.replace("prediction", "\\text{prediction}")
            content = content.replace("value_pred", "\\text{value\\_pred}")
            content = content.replace("value_target", "\\text{value\\_target}")
            content = content.replace("return", "\\text{return}")
            content = content.replace("total_loss", "\\text{total\\_loss}")
            content = content.replace("policy_loss", "\\text{policy\\_loss}")
            content = content.replace("vf_loss", "\\text{vf\\_loss}")
            content = content.replace("entropy_loss", "\\text{entropy\\_loss}")
            content = content.replace("vf_coeff", "\\text{vf\\_coeff}")
            content = content.replace("value_head", "\\text{value\\_head}")
            content = content.replace("gain", "\\text{gain}")
            content = content.replace("bias", "\\text{bias}")
            content = content.replace("orthogonal init", "\\text{orthogonal init}")
            
            # Handle fraction if it looks like a/b
            content = re.sub(r"(\w+)\s*/\s*(\w+)", r"\\frac{\1}{\2}", content)
            
            return f"$$\n{content}\n$$"
        return match.group(0)

    text = re.sub(r"```(text|)\n(.*?)\n```", replace_code_block, text, flags=re.DOTALL)

    # 2. Convert specific inline patterns
    # V(s) ≈ b -> $V(s) \approx b$
    text = re.sub(r"`V\(s\) ≈ (.*?)`", r"$V(s) \approx \1$", text)
    text = re.sub(r"`gain=(.*?)`", r"$\text{gain}=\1$", text)
    text = re.sub(r"`bias=(.*?)`", r"$\text{bias}=\1$", text)
    text = re.sub(r"`\+40 - 0.03 × 100 steps ≈ \+37`", r"$+40 - 0.03 \times 100 \text{ steps} \approx +37$", text)
    text = re.sub(r"`-5 - 0.03 × 50 steps ≈ -6.5`", r"$-5 - 0.03 \times 50 \text{ steps} \approx -6.5$", text)
    text = re.sub(r"`\+40 -5 - 0.03 × 200 steps ≈ \+29`", r"$+40 - 5 - 0.03 \times 200 \text{ steps} \approx +29$", text)
    
    # Simple variables
    inline_vars = ["V(s)", "value_pred", "value_target", "vf_loss", "vf_coeff", "value_head", "VE", "MSE", "ReLU(x)"]
    for var in inline_vars:
        safe_var = var.replace("(", "\\(").replace(")", "\\)")
        text = re.sub(f"`{safe_var}`", f"${var.replace('_', '\\_')}$", text)

    # Specific ranges like -10 ~ 200
    text = re.sub(r"`(-?\d+)\s*~\s*(-?\d+)`", r"$\1 \\sim \2$", text)

    return text

dir_path = "/home/aa/Documents/Obsidian Vault/消融前"
for filename in os.listdir(dir_path):
    if filename.endswith(".md"):
        file_path = os.path.join(dir_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        new_content = convert_to_latex(content)
        
        if new_content != content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"Updated: {filename}")
        else:
            print(f"No changes: {filename}")
