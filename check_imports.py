import ast
import os

def check_imports(file_path):
    with open(file_path, "r") as f:
        tree = ast.parse(f.read(), filename=file_path)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("app."):
                mod_path = node.module.replace(".", "/") + ".py"
                if not os.path.exists(mod_path):
                    print(f"[{file_path}] MISSING MODULE: {node.module}")
                else:
                    with open(mod_path, "r") as mf:
                        mod_tree = ast.parse(mf.read())
                    names_in_mod = {n.name for n in ast.walk(mod_tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))}
                    names_in_mod.update({t.id for t in ast.walk(mod_tree) if isinstance(t, ast.Name) and isinstance(getattr(t, "ctx", None), ast.Store)})
                    
                    for alias in node.names:
                        if alias.name not in names_in_mod and alias.name != "*":
                            print(f"[{file_path}] Name {alias.name} NOT FOUND in {node.module}")

import glob
for py_file in glob.glob("app/**/*.py", recursive=True):
    try:
        check_imports(py_file)
    except Exception as e:
        print(f"Error checking {py_file}: {e}")

print("Done checking imports.")
