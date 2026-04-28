import gradio as gr
from test_agent import update_knowledge_with_new_paper, load_paper_informaion

def learn_new_paper(additional_input: str = ""):
    result = update_knowledge_with_new_paper(additional_input)
    return result["title"], result["explanation"], result["paper_base"]

with gr.Blocks() as app_demo:
    
    with gr.Tab("Learn New Paper"):
        gr.Markdown("# Paper Learning Agent")
        with gr.Row():
            additional_input = gr.Textbox(label="Do you want to learn about anything specific next?")
        with gr.Row():
            paper_button = gr.Button("Learn new paper")
        with gr.Row():
            paper_title = gr.Textbox(label="Paper Title")
        with gr.Row():
            paper_explanation = gr.Textbox(label="Paper Explanation")
        with gr.Row():
            paper_base_new = gr.Textbox(label="Paper Base")
        paper_button.click(learn_new_paper, inputs=[additional_input], outputs=[paper_title, paper_explanation, paper_base_new])
    
    with gr.Tab("Current Paper Base"):
        gr.Markdown("# Current Paper Base")
        with gr.Row():
            paper_base = gr.Textbox(label="Paper Base")
    with gr.Tab("Current Paper List"):
        gr.Markdown("# Current Paper List")
        with gr.Row():
            paper_list = gr.Textbox(label="Paper List")
    with gr.Tab("Current Paper Goals"):
        gr.Markdown("# Current Paper Goals")
        with gr.Row():
            knowledge_base = gr.Textbox(label="Paper Goals")
    app_demo.load(load_paper_informaion, outputs=[knowledge_base, paper_base, paper_list])                  


app_demo.launch()