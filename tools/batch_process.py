import glob
import os
import subprocess
import sys

# Ensure project root is on sys.path so `from backend...` works when running this script
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.image_processing import preprocess_image

if __name__ == '__main__':
    processed_files = []
    for p in sorted(glob.glob(os.path.join('uploads', '*'))):
        try:
            out = preprocess_image(p)
            print('PROCESSED', out)
            processed_files.append(out)
        except Exception as e:
            print('ERROR processing', p, ':', e)

    if not processed_files:
        print('No files processed.')
    else:
        # choose newest processed file
        newest = max(processed_files, key=os.path.getmtime)
        print('Running layout extraction on', newest)
        subprocess.run(['python', 'layout_extraction.py', newest])
