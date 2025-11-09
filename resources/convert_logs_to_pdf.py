"""
Convert interaction logs (JSONL) to formatted PDF conversation transcripts.

Usage:
    python convert_logs_to_pdf.py <input_file.jsonl> [output_file.pdf]

Example:
    python convert_logs_to_pdf.py interaction_logs/adaptive_edac04f9.jsonl
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file and return list of interactions."""
    interactions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    interactions.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed line: {e}")
                    continue
    return interactions


def create_pdf(interactions: List[Dict], output_path: str):
    """Create a formatted PDF from interaction logs."""

    # Create PDF document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch,
    )

    # Container for the PDF elements
    story = []

    # Define styles
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2C3E50'),
        spaceAfter=30,
        alignment=TA_CENTER,
    )

    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.HexColor('#7F8C8D'),
        spaceAfter=20,
        alignment=TA_CENTER,
    )

    student_style = ParagraphStyle(
        'Student',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#2C3E50'),
        leftIndent=20,
        rightIndent=40,
        spaceAfter=8,
        spaceBefore=4,
    )

    agent_style = ParagraphStyle(
        'Agent',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#34495E'),
        leftIndent=40,
        rightIndent=20,
        spaceAfter=8,
        spaceBefore=4,
    )

    label_style = ParagraphStyle(
        'Label',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#95A5A6'),
        spaceAfter=2,
    )

    timestamp_style = ParagraphStyle(
        'Timestamp',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#BDC3C7'),
        spaceAfter=2,
    )

    metadata_style = ParagraphStyle(
        'Metadata',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#BDC3C7'),
        alignment=TA_CENTER,
    )

    # Get session metadata from first interaction
    if not interactions:
        print("No interactions found in file")
        return

    first = interactions[0]
    session_id = first.get('session_id', 'Unknown')
    condition = first.get('condition', 'Unknown').upper()
    total_interactions = len([i for i in interactions if i.get('student_message')])

    # Calculate session duration
    timestamps = [i.get('timestamp') for i in interactions if i.get('timestamp')]
    if timestamps:
        start_time = datetime.fromisoformat(timestamps[0])
        end_time = datetime.fromisoformat(timestamps[-1])
        duration = end_time - start_time
        duration_str = f"{duration.total_seconds() / 60:.1f} minutes"
        start_str = start_time.strftime("%B %d, %Y at %I:%M %p")
    else:
        duration_str = "Unknown"
        start_str = "Unknown"

    # Count completed steps
    completed_steps = set()
    for interaction in interactions:
        if interaction.get('step_completed'):
            completed_steps.add(interaction.get('current_step'))

    # Title page
    story.append(Paragraph("Giro Tutoring Session", title_style))
    story.append(Paragraph(f"Session ID: {session_id}", subtitle_style))

    # Session info table
    session_info = [
        ["Condition:", condition],
        ["Started:", start_str],
        ["Duration:", duration_str],
        ["Total Interactions:", str(total_interactions)],
        ["Steps Completed:", ", ".join(map(str, sorted(completed_steps))) or "None"],
    ]

    info_table = Table(session_info, colWidths=[2*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'Helvetica', 10),
        ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 10),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#7F8C8D')),
        ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#2C3E50')),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))

    story.append(info_table)
    story.append(Spacer(1, 0.5*inch))

    # Add conversation
    story.append(Paragraph("Conversation Transcript", styles['Heading2']))
    story.append(Spacer(1, 0.3*inch))

    current_step = None

    for idx, interaction in enumerate(interactions, 1):
        student_msg = interaction.get('student_message', '').strip()
        agent_msg = interaction.get('agent_response', '').strip()

        # Skip empty interactions
        if not student_msg and not agent_msg:
            continue

        # Add step header if step changed
        step = interaction.get('current_step')
        if step and step != current_step:
            current_step = step
            story.append(Spacer(1, 0.2*inch))
            story.append(Paragraph(
                f"<b>Exercise {step}</b>",
                ParagraphStyle(
                    'StepHeader',
                    parent=styles['Heading3'],
                    fontSize=14,
                    textColor=colors.HexColor('#3498DB'),
                    spaceAfter=10,
                )
            ))

        # Interaction container
        interaction_elements = []

        # Format timestamp
        timestamp_str = ""
        if interaction.get('timestamp'):
            try:
                ts = datetime.fromisoformat(interaction['timestamp'])
                timestamp_str = ts.strftime("%I:%M:%S %p")
            except:
                timestamp_str = ""

        # Student message
        if student_msg:
            label_text = f"<font color='#95A5A6'><b>Student:</b></font>"
            if timestamp_str:
                label_text += f" <font color='#D5DBDB'><i>{timestamp_str}</i></font>"
            interaction_elements.append(Paragraph(label_text, label_style))
            interaction_elements.append(Paragraph(
                student_msg.replace('\n', '<br/>'),
                student_style
            ))

        # Agent message
        if agent_msg:
            label_text = f"<font color='#3498DB'><b>Tutor (GIRO):</b></font>"
            if timestamp_str:
                label_text += f" <font color='#D5DBDB'><i>{timestamp_str}</i></font>"
            interaction_elements.append(Paragraph(label_text, label_style))
            interaction_elements.append(Paragraph(
                agent_msg.replace('\n', '<br/>'),
                agent_style
            ))

        # Add metadata as small text
        meta_parts = []
        if interaction.get('hints_given', 0) > 0:
            meta_parts.append(f"Hints: {interaction['hints_given']}")
        if interaction.get('step_completed'):
            meta_parts.append("✓ Step Completed")
        if interaction.get('difficulty_adjusted'):
            meta_parts.append(f"Difficulty: {interaction.get('difficulty_level', '').upper()}")

        if meta_parts:
            interaction_elements.append(Paragraph(
                " | ".join(meta_parts),
                metadata_style
            ))

        interaction_elements.append(Spacer(1, 0.15*inch))

        # Keep interaction together on same page
        story.append(KeepTogether(interaction_elements))

    # Footer
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        metadata_style
    ))

    # Build PDF
    doc.build(story)
    print(f"✓ PDF created: {output_path}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python convert_logs_to_pdf.py <input_file.jsonl> [output_file.pdf]")
        print("\nExample:")
        print("  python convert_logs_to_pdf.py interaction_logs/adaptive_edac04f9.jsonl")
        sys.exit(1)

    input_file = sys.argv[1]

    # Generate output filename if not provided
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        input_path = Path(input_file)
        output_file = input_path.with_suffix('.pdf').name

    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    print(f"Loading interactions from {input_file}...")
    interactions = load_jsonl(input_file)
    print(f"Found {len(interactions)} interactions")

    print(f"Creating PDF: {output_file}...")
    create_pdf(interactions, output_file)

    print(f"\n✓ Successfully created {output_file}")


if __name__ == "__main__":
    main()