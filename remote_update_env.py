import os
import re

env_path = '/home/azureuser/walletbot/.env'
if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        content = f.read()
    
    # Relax confidence to 0
    content = re.sub(r'ML_CONFIDENCE_THRESHOLD=.*', 'ML_CONFIDENCE_THRESHOLD=0.0', content)
    # Disable ensemble models to force alpha-only
    content = re.sub(r'ENSEMBLE_ACTIVE_MODELS=.*', 'ENSEMBLE_ACTIVE_MODELS=', content)
    # Also ensure execution mode is robust_paper if it's not
    if 'EXECUTION_MODE=' not in content:
        content += '\nEXECUTION_MODE=robust_paper'
    else:
        content = re.sub(r'EXECUTION_MODE=.*', 'EXECUTION_MODE=robust_paper', content)

    with open(env_path, 'w') as f:
        f.write(content)
    print("Updated .env")
else:
    print(".env not found at " + env_path)
