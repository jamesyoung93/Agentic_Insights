"""
Generate Architecture Overview PowerPoint Slide (Fixed)
Simplified and more robust approach
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
import json

# Load architecture data
with open('architecture_graph.json', 'r') as f:
    arch_data = json.load(f)

# Create presentation (16:9 format)
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# Add blank slide
blank_layout = prs.slide_layouts[6]  # Blank layout
slide = prs.slides.add_slide(blank_layout)

# Color palette (grayscale + blue accent)
COLOR_TITLE = RGBColor(31, 56, 100)
COLOR_DATASTORE = RGBColor(176, 196, 222)  # Light steel blue
COLOR_SERVICE = RGBColor(144, 164, 174)    # Blue grey
COLOR_EXTERNAL = RGBColor(255, 152, 0)     # Orange accent
COLOR_OUTPUT = RGBColor(189, 189, 189)     # Grey
COLOR_TEXT = RGBColor(33, 33, 33)
COLOR_EDGE = RGBColor(66, 66, 66)

# Fonts
FONT_NAME = 'Calibri'

# Grid layout (6 columns)
MARGIN_LEFT = Inches(0.4)
MARGIN_TOP = Inches(1.3)
COL_WIDTH = Inches(1.9)
COL_SPACING = Inches(0.2)
ROW_HEIGHT = Inches(0.7)
ROW_SPACING = Inches(0.35)

# ============================================================================
# TITLE & CAPTION
# ============================================================================
title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12), Inches(0.5))
title_frame = title_box.text_frame
title_frame.text = "Data to Discovery: System Architecture"
title_para = title_frame.paragraphs[0]
title_para.font.name = FONT_NAME
title_para.font.size = Pt(36)
title_para.font.bold = True
title_para.font.color.rgb = COLOR_TITLE
title_para.alignment = PP_ALIGN.LEFT

caption_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.8), Inches(12), Inches(0.3))
caption_frame = caption_box.text_frame
caption_frame.text = "Autonomous discovery through statistical analysis, agent orchestration, and LLM synthesis"
caption_para = caption_frame.paragraphs[0]
caption_para.font.name = FONT_NAME
caption_para.font.size = Pt(16)
caption_para.font.italic = True
caption_para.font.color.rgb = COLOR_TEXT
caption_para.alignment = PP_ALIGN.LEFT

# ============================================================================
# NODE POSITIONING
# ============================================================================
node_positions = {}

# Define all nodes with (id, col, row, label)
all_nodes = [
    # Sources (Column 0)
    ('csv_data', 0, 0, "CSV\nData"),
    ('literature_store', 0, 1, "Literature\nStore"),
    ('config', 0, 2, "Config"),

    # Processing (Column 1)
    ('streamlit_ui', 1, 0, "Streamlit\nUI"),
    ('kosmos_framework', 1, 2, "Kosmos\nCLI"),

    # Agents (Column 2)
    ('data_analyst', 2, 0, "Data\nAnalyst"),
    ('lit_search_agent', 2, 1, "Literature\nAgent"),
    ('world_model', 2, 2, "World\nModel"),

    # External (Column 3)
    ('openai_api', 3, 1, "OpenAI\nAPI"),

    # Storage (Column 4)
    ('world_model_state', 4, 2, "State\nJSON"),

    # Outputs (Column 5)
    ('enhanced_report', 5, 1, "Report\nTXT"),
    ('analysis_artifacts', 5, 2, "Analysis\nCode"),
]

# Color mapping function
def get_node_color(node_type):
    if node_type == 'datastore':
        return COLOR_DATASTORE
    elif node_type == 'service':
        return COLOR_SERVICE
    elif node_type == 'external':
        return COLOR_EXTERNAL
    elif node_type in ['input', 'output']:
        return COLOR_OUTPUT
    else:
        return COLOR_SERVICE

# Draw nodes
shapes = {}
for node_id, col, row, label in all_nodes:
    # Find node data
    node_data = next((n for n in arch_data['nodes'] if n['id'] == node_id), None)
    if not node_data:
        continue

    # Calculate position
    left = MARGIN_LEFT + col * (COL_WIDTH + COL_SPACING)
    top = MARGIN_TOP + row * (ROW_HEIGHT + ROW_SPACING)
    width = COL_WIDTH
    height = ROW_HEIGHT

    # Store center position for connectors
    node_positions[node_id] = {
        'left': left,
        'top': top,
        'width': width,
        'height': height,
        'center_x': left + width / 2,
        'center_y': top + height / 2
    }

    # Determine shape
    shape_type_name = node_data.get('shape', 'rectangle')
    if shape_type_name == 'cylinder' or node_data.get('type') == 'datastore':
        shape_type = MSO_SHAPE.CAN
    elif shape_type_name == 'hexagon' or node_data.get('type') == 'external':
        shape_type = MSO_SHAPE.HEXAGON
    elif shape_type_name == 'parallelogram' or node_data.get('type') in ['input', 'output']:
        shape_type = MSO_SHAPE.PARALLELOGRAM
    else:
        shape_type = MSO_SHAPE.ROUNDED_RECTANGLE

    # Add shape
    shape = slide.shapes.add_shape(shape_type, left, top, width, height)

    # Fill color
    fill_color = get_node_color(node_data.get('type', 'service'))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color

    # Border
    shape.line.color.rgb = COLOR_EDGE
    shape.line.width = Pt(1.5)

    # Text
    text_frame = shape.text_frame
    text_frame.text = label
    text_frame.word_wrap = True
    text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE

    para = text_frame.paragraphs[0]
    para.font.name = FONT_NAME
    para.font.size = Pt(16)
    para.font.bold = True
    para.font.color.rgb = COLOR_TEXT
    para.alignment = PP_ALIGN.CENTER

    shapes[node_id] = shape

# ============================================================================
# CONNECTORS (Arrows)
# ============================================================================
# Key edges to display
key_edges = [
    ('csv_data', 'streamlit_ui'),
    ('config', 'streamlit_ui'),
    ('streamlit_ui', 'data_analyst'),
    ('streamlit_ui', 'world_model'),
    ('data_analyst', 'openai_api'),
    ('data_analyst', 'world_model'),
    ('lit_search_agent', 'openai_api'),
    ('lit_search_agent', 'world_model'),
    ('world_model', 'world_model_state'),
    ('world_model', 'enhanced_report'),
    ('data_analyst', 'analysis_artifacts'),
    ('literature_store', 'lit_search_agent'),
]

for from_id, to_id in key_edges:
    if from_id not in node_positions or to_id not in node_positions:
        continue

    from_pos = node_positions[from_id]
    to_pos = node_positions[to_id]

    # Calculate connector endpoints (from right edge to left edge)
    from_x = from_pos['left'] + from_pos['width']
    from_y = from_pos['center_y']
    to_x = to_pos['left']
    to_y = to_pos['center_y']

    # Add straight connector
    try:
        connector = slide.shapes.add_connector(
            1,  # msoConnectorStraight
            from_x, from_y,
            to_x, to_y
        )
        connector.line.color.rgb = COLOR_EDGE
        connector.line.width = Pt(1.5)
    except Exception as e:
        print(f"Warning: Could not add connector from {from_id} to {to_id}: {e}")

# ============================================================================
# LEGEND
# ============================================================================
legend_left = Inches(10.5)
legend_top = Inches(5.8)
legend_width = Inches(2.5)

# Legend background box (optional, makes it stand out)
legend_bg = slide.shapes.add_shape(
    MSO_SHAPE.ROUNDED_RECTANGLE,
    legend_left - Inches(0.1),
    legend_top - Inches(0.1),
    legend_width + Inches(0.2),
    Inches(1.4)
)
legend_bg.fill.solid()
legend_bg.fill.fore_color.rgb = RGBColor(245, 245, 245)
legend_bg.line.color.rgb = RGBColor(200, 200, 200)
legend_bg.line.width = Pt(0.5)

# Legend title
legend_title = slide.shapes.add_textbox(legend_left, legend_top, legend_width, Inches(0.25))
legend_title_frame = legend_title.text_frame
legend_title_frame.text = "Legend"
legend_title_para = legend_title_frame.paragraphs[0]
legend_title_para.font.name = FONT_NAME
legend_title_para.font.size = Pt(16)
legend_title_para.font.bold = True
legend_title_para.font.color.rgb = COLOR_TEXT

# Legend items
legend_items = [
    ("● Datastore", COLOR_DATASTORE),
    ("● Service/Module", COLOR_SERVICE),
    ("● External API", COLOR_EXTERNAL),
    ("● Input/Output", COLOR_OUTPUT),
]

legend_y = legend_top + Inches(0.35)
for item_text, item_color in legend_items:
    item_box = slide.shapes.add_textbox(legend_left, legend_y, legend_width, Inches(0.22))
    item_frame = item_box.text_frame
    item_frame.text = item_text
    item_para = item_frame.paragraphs[0]
    item_para.font.name = FONT_NAME
    item_para.font.size = Pt(13)

    # Color the bullet
    if item_para.runs:
        item_para.runs[0].font.color.rgb = item_color
        item_para.runs[0].font.size = Pt(16)
        item_para.runs[0].font.bold = True

        for run in item_para.runs[1:]:
            run.font.color.rgb = COLOR_TEXT

    legend_y += Inches(0.22)

# ============================================================================
# SAVE
# ============================================================================
output_path = 'architecture_overview.pptx'
prs.save(output_path)

print(f"✅ PowerPoint slide created: {output_path}")
print(f"   - Nodes: {len(all_nodes)} (constraint: ≤12)")
print(f"   - Edges shown: {len(key_edges)} (constraint: ≤16)")
print(f"   - Format: 16:9 (13.333\" × 7.5\")")
print(f"   - Style: Consulting-grade, diagram-first")
print(f"   - File size: {len(open(output_path, 'rb').read()) / 1024:.1f} KB")
