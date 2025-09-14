from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# =============================
# Generate Sample Medical Report
# =============================
file_path = "sample_medical_report2.pdf"

doc = SimpleDocTemplate(file_path, pagesize=A4)
styles = getSampleStyleSheet()
story = []

# Title
story.append(Paragraph("Apollo Hospitals - Patient Diagnostic Report", styles["Title"]))
story.append(Spacer(1, 12))

# Patient info
story.append(Paragraph("<b>Patient Name:</b> Ravi Kumar", styles["Normal"]))
story.append(Paragraph("<b>Age:</b> 45", styles["Normal"]))
story.append(Paragraph("<b>Gender:</b> Male", styles["Normal"]))
story.append(Spacer(1, 12))

# Reported Complaints
story.append(Paragraph("<b>Reported Complaints</b>", styles["Heading2"]))
complaints = """
Patient reports persistent <b>fever</b> for the past 5 days along with <b>dry cough</b>, 
occasional <b>shortness of breath</b>, and <b>chest pain</b>. 
There is also history of <b>fatigue</b> and <b>loss of appetite</b>.
"""
story.append(Paragraph(complaints, styles["Normal"]))
story.append(Spacer(1, 12))

# Lab findings table
story.append(Paragraph("<b>Lab Investigations</b>", styles["Heading2"]))
table_data = [
    ["Test", "Result", "Normal Range"],
    ["Hemoglobin", "11.2 g/dL", "13.5 - 17.5 g/dL"],
    ["WBC Count", "12,000 /µL", "4,000 - 10,000 /µL"],
    ["Blood Sugar (Fasting)", "140 mg/dL", "70 - 110 mg/dL"],
    ["Chest X-Ray", "Mild lung infection", "Normal"]
]
table = Table(table_data, hAlign="LEFT")
table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
]))
story.append(table)
story.append(Spacer(1, 12))

# Doctor Note
story.append(Paragraph("<b>Doctor's Note:</b>", styles["Heading2"]))
note = """
Findings are suggestive of a possible lower respiratory tract infection. 
Recommend antibiotics, supportive care, and further evaluation if symptoms persist.
"""
story.append(Paragraph(note, styles["Normal"]))

# Build PDF
doc.build(story)

print(f"✅ Sample report saved as {file_path}")
