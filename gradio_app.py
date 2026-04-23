import gradio as gr
from test_agent import update_knowledge_with_new_paper, load_paper_informaion

def learn_new_paper(additional_input: str = ""):
    result = update_knowledge_with_new_paper(additional_input)
    return result["title"], result["explanation"], result["paper_base"]

with gr.Blocks() as app_demo:
    gr.Markdown("# Paper Learning Agent")
    with gr.Row():
        paper_button = gr.Button("Learn new paper")
    with gr.Row():
        paper_title = gr.Textbox(label="Paper Title")
    with gr.Row():
        paper_explanation = gr.Textbox(label="Paper Explanation")
    with gr.Row():
        paper_base = gr.Textbox(label="Paper Base")
    paper_button.click(learn_new_paper, inputs=None, outputs=[paper_title, paper_explanation, paper_base])


app_demo.launch()