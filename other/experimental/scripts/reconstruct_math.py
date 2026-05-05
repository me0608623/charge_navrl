import os

def final_reconstruct(text):
    # 1. Emergency Clean: Fix all corrupted \text{ artifacts
    # These are caused by \t being interpreted as Tab
    text = text.replace("    ext{", "\\text{")
    text = text.replace("   ext{", "\\text{")
    text = text.replace("  ext{", "\\text{")
    text = text.replace(" ext{", "\\text{")
    text = text.replace("\text{", "\\text{")
    
    # Special fix for the most common corruption
    text = text.replace("\t", "\\t") # Restore literal \t to \\t
    text = text.replace("\\text{value}_{\\text{loss}", "\\text{value\\_loss}")
    text = text.replace("\\text{vf}_{\\text{coeff}", "\\text{vf\\_coeff}")
    text = text.replace("\\text{vf}_{\\text{loss}", "\\text{vf\\_loss}")
    text = text.replace("\\text{target}_mean", "\\text{target}_{\\text{mean}}")
    text = text.replace("\\text{value\\_pred}_mean", "\\text{value\\_pred}_{\\text{mean}}")
    
    # Define the target mathematical blocks that should be corrected
    # Section 1: Returns calculation
    old_block1 = """```text
好的 episode: +40 - 0.03 × 100 steps ≈ +37
壞的 episode: -5 - 0.03 × 50 steps ≈ -6.5
混合 episode: +40 -5 - 0.03 × 200 steps ≈ +29
```"""
    new_block1 = """$$
\\text{好的 episode: } +40 - 0.03 \\times 100 \\text{ steps} \\approx +37 \\\\
\\text{壞的 episode: } -5 - 0.03 \\times 50 \\text{ steps} \\approx -6.5 \\\\
\\text{混合 episode: } +40 - 5 - 0.03 \\times 200 \\text{ steps} \\approx +29
$$"""
    text = text.replace(old_block1, new_block1)

    # Correct common variables everywhere (using $...$ to ensure it's math context)
    replacements = {
        "value_loss": "\\text{value\\_loss}",
        "vf_loss": "\\text{vf\\_loss}",
        "policy_loss": "\\text{policy\\_loss}",
        "entropy_loss": "\\text{entropy\\_loss}",
        "total_loss": "\\text{total\\_loss}",
        "vf_coeff": "\\text{vf\\_coeff}",
        "value_pred": "\\text{value\\_pred}",
        "value_target": "value_{\\text{target}}",
        "normalized_return": "\\text{normalized\\_return}"
    }

    for k, v in replacements.items():
        # Correct $var_name$ or $var\_name$
        text = text.replace(f"${k}$", f"${v}$")
        text = text.replace(f"${k.replace('_', r'\_')}$", f"${v}$")
    
    # Final cleanup of double text
    text = text.replace("\\text{\\text{", "\\text{")
    text = text.replace("}}", "}")
    
    # Fix the MSE formula block specifically
    old_mse = """$$
        ext{value}_{    ext{loss} = \operatorname{MSE}(\text{value\_pred}, value_\text{target})
           = \operatorname{mean}((\text{value\_pred} - value_\text{target})^2)
$$"""
    new_mse = """$$
\\text{value\\_loss} = \\operatorname{MSE}(\\text{value\\_pred}, \\text{value\\_target})
           = \\operatorname{mean}((\\text{value\\_pred} - \\text{value\\_target})^2)
$$"""
    text = text.replace(old_mse, new_mse)

    return text

file_path = "/home/aa/Documents/Obsidian Vault/消融前/02_RL基礎_Returns-GAE與NormalizeReturn.md"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

new_content = final_reconstruct(content)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(new_content)

print("Reconstructed with high precision.")
