from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

import sys
from pathlib import Path

if len(sys.argv) < 3:
    print('Usage: python tools/generate_pdf.py input_md output_pdf')
    sys.exit(2)

input_md = Path(sys.argv[1])
output_pdf = Path(sys.argv[2])

text = input_md.read_text(encoding='utf-8')
lines = text.splitlines()

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='MyHeading1', parent=styles['Heading1'], fontSize=16, leading=18, spaceAfter=12))
styles.add(ParagraphStyle(name='MyHeading2', parent=styles['Heading2'], fontSize=13, leading=15, spaceAfter=8))
styles.add(ParagraphStyle(name='MyBody', parent=styles['Normal'], fontSize=10, leading=12))

flowables = []

for line in lines:
    if line.strip().startswith('# '):
        flowables.append(Paragraph(line.strip('# ').strip(), styles['MyHeading1']))
    elif line.strip().startswith('## '):
        flowables.append(Paragraph(line.strip('# ').strip(), styles['MyHeading2']))
    elif line.strip().startswith('---') or line.strip().startswith('==='):
        flowables.append(Spacer(1, 6))
    elif line.strip() == '':
        flowables.append(Spacer(1, 6))
    else:
        # escape < and > which can break simple XML in Paragraph
        safe = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        flowables.append(Paragraph(safe, styles['MyBody']))

# Build PDF
output_pdf.parent.mkdir(parents=True, exist_ok=True)
doc = SimpleDocTemplate(str(output_pdf), pagesize=letter,
                        leftMargin=0.7*inch, rightMargin=0.7*inch,
                        topMargin=0.75*inch, bottomMargin=0.75*inch)

doc.build(flowables)
print('WROTE', output_pdf)
