import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

def create_lab_pdf():
    pdf_filename = "A_Star_Lab_Assignment.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        rightMargin=54, leftMargin=54,
        topMargin=54, bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Styles for Clean Design
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#2c3e50'),
        alignment=TA_CENTER,
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#2980b9'),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#333333'),
        spaceAfter=8
    )
    
    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Code'],
        fontName='Courier',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#2c3e50'),
        backColor=colors.HexColor('#f8f9fa'),
        borderColor=colors.HexColor('#e2e8f0'),
        borderWidth=1,
        borderPadding=6,
        spaceAfter=8
    )

    story = []
    
    # Header Elements
    story.append(Paragraph("🕒 The 30-Minute A* Stress-Test & Heuristic Challenge", title_style))
    story.append(Spacer(1, 10))
    
    # Objectives
    story.append(Paragraph("Learning Objectives:", h1_style))
    story.append(Paragraph("• <b>Testing & QA:</b> Learn how to write black-box adversarial test cases to expose algorithmic bias.", body_style))
    story.append(Paragraph("• <b>Algorithm Adaptation:</b> Modify search behavior by implementing a new geometric heuristic function.", body_style))
    story.append(Spacer(1, 10))
    
    # Part 1
    story.append(Paragraph("🛠️ Part 1: The Heuristic Code Injection (10 Minutes)", h1_style))
    story.append(Paragraph(
        "Open your <code>environment.py</code> file. Your task is to write the mathematical formula for "
        "<b>Chebyshev Distance</b>. Chebyshev distance models 8-way movement where diagonal steps cost "
        "the same as cardinal steps (cost = 1).", body_style
    ))
    
    story.append(Paragraph("Task 1.1: Add this method to your <code>SurgicalLabEnvironment</code> class:", body_style))
    code_text_1 = (
        "def chebyshev_distance(self, x1: int, y1: int, x2: int, y2: int) -> float:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;\"\"\"Calculates 8-way Chebyshev distance\"\"\"<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;# TODO: Implement the single-line Chebyshev math here<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;return max(abs(x1 - x2), abs(y1 - y2))"
    )
    story.append(Paragraph(code_text_1, code_style))
    
    story.append(Paragraph("Task 1.2: Update the heuristic checks inside your <code>planner.py</code>:", body_style))
    code_text_2 = (
        "if heuristic_type == 'manhattan':<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;start_node.h = self.env.manhattan_distance(start[0], start[1], goal[0], goal[1])<br/>"
        "elif heuristic_type == 'chebyshev':<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;start_node.h = self.env.chebyshev_distance(start[0], start[1], goal[0], goal[1])<br/>"
        "else:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;start_node.h = self.env.euclidean_distance(start[0], start[1], goal[0], goal[1])"
    )
    story.append(Paragraph(code_text_2, code_style))
    story.append(Spacer(1, 10))
    
    # Part 2
    story.append(Paragraph("🎨 Part 2: Adversarial Map Design (10 Minutes)", h1_style))
    story.append(Paragraph(
        "Boot up the <b>HTML Visual Map Editor</b> in your browser. Design a specific 20x15 obstacle layout "
        "(under 30% density) that causes <b>Manhattan Distance</b> to perform terribly (expand significantly more nodes), "
        "while Euclidean distance and/or  your newly coded <b>Chebyshev Distance</b> breezes through it with optimal efficiency. Export your file as <code>config.py</code>.", body_style
    ))
    story.append(Spacer(1, 10))
    
    # Part 3
    story.append(Paragraph("📊 Part 3: The Benchmark & QA Analysis (10 Minutes)", h1_style))
    story.append(Paragraph("Run your experiment suite in the terminal (<code>python main.py</code>) using your custom map profile and fill out the report tables below.", body_style))
    story.append(Spacer(1, 5))
    
    # Metrics Table Matrix Setup
    table_data = [
        [Paragraph('<b>Heuristic</b>', body_style), Paragraph('<b>Path Length Found</b>', body_style), Paragraph('<b>Total Nodes Expanded</b>', body_style), Paragraph('<b>Execution Time (ms)</b>', body_style)],
        [Paragraph('<b>Manhattan</b>', body_style), '', '', ''],
        [Paragraph('<b>Euclidean</b>', body_style), '', '', ''],
        [Paragraph('<b>Chebyshev</b>', body_style), '', '', '']
    ]
    metrics_table = Table(table_data, colWidths=[120, 120, 130, 134], rowHeights=[24, 28, 28, 28])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#ecf0f1')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#bdc3c7')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 15))
    
    # Short Answer Questions
    story.append(Paragraph("<b>Q1. The 'Why':</b> Why did Manhattan over-explore so many extra nodes compared to Euclidean/Chebyshev? What specific pattern of walls exploited Manhattan's grid-locked math?", body_style))
    story.append(Spacer(1, 40)) # Space for writing
    
    story.append(Paragraph("<b>Q2. Heuristic Invalidation:</b> Even though Chebyshev expands fewer nodes, your current <code>get_neighbors()</code> function only allows 4-way cardinal movement. Because your forklift cannot actually move diagonally, does Chebyshev remain an <b>admissible heuristic</b>? Explain the risk of utilizing it here.", body_style))
    story.append(Spacer(1, 40)) # Space for writing
    
    # Build Document
    doc.build(story)
    print(f"Success! Generated clean assignment sheet: '{os.path.abspath(pdf_filename)}'")

if __name__ == '__main__':
    create_lab_pdf()
