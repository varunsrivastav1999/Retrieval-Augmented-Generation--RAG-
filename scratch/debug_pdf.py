import fitz
import sys

pdf_path = "/Users/tei-1358/Documents/i-Tips Projects/TM-2000_TVS/Dev/external_media/default/sh080809engx.pdf"
doc = fitz.open(pdf_path)
found = False

print(f"Searching for 'Gratis' or 'Warranty' in {pdf_path}...")

for i in range(len(doc)):
    text = doc[i].get_text()
    if "Gratis" in text or "Warranty" in text:
        print(f"Found on Page {i+1}:")
        print(text[:200])
        found = True

if not found:
    print("Not found in text. Checking images...")
    for i in range(len(doc)):
        images = doc[i].get_images()
        if images:
            print(f"Page {i+1} has {len(images)} images.")

doc.close()
