import gradio as gr
from utils.file_utils import load_file
from src.agent.graph import graph
import uuid

def learn_new_paper(learning_preferences: str = "", complexity_preference: str = "medium") -> str:
    id = uuid.uuid4()
    config = {"configurable": {"thread_id": str(id)}}
    agent_state = {"messages": [], "query_preference": learning_preferences, "complexity_preferences": complexity_preference}
    result = graph.invoke(agent_state, config=config)
    concept_base = load_file("src/data/concept_base.txt", "No concepts learned yet")
    return result["paper_title"], result["paper_concept"], result["paper_summary"], concept_base

def load_data_files():
    from utils.file_utils import load_file
    return (
        load_file("src/data/research_goals.txt", "No knowledge base found"),
        load_file("src/data/concept_base.txt", "No concepts analyzed yet"),
        load_file("src/data/paper_list.txt", "No papers analyzed yet"),
    )

def reset_paper_list():
    from utils.file_utils import save_file
    save_file("src/data/paper_list.txt", "")
    return ""

def update_research_goals(new_research_goals: str):
    from utils.file_utils import save_file
    save_file("src/data/research_goals.txt", new_research_goals)
    return new_research_goals

with gr.Blocks() as app_demo:
    
    with gr.Tab("Learn New Concept"):
        gr.Markdown("# Paper Learning Agent")
        with gr.Row():
            learning_input = gr.Textbox(label="Do you want to learn about anything specific next?")
        with gr.Row():
            complexity_input = gr.Textbox(label="Complexity (beginner / medium / advanced)")
        with gr.Row():
            concept_button = gr.Button("Learn new concept")
        with gr.Row():
            paper_title = gr.Textbox(label="Paper Title")
        with gr.Row():
            paper_concept = gr.Textbox(label="New learned Concept")
        with gr.Row():
            paper_summary = gr.Textbox(label="Paper Summary")
        with gr.Row():
            new_concept_base = gr.Textbox(label="New Concept Base")
        concept_button.click(learn_new_paper, inputs=[learning_input,complexity_input], outputs=[paper_title, paper_concept, paper_summary, new_concept_base])
    
    with gr.Tab("Current Concept Base"):
        with gr.Row():
            concept_base = gr.Textbox(label="Paper Base")
    with gr.Tab("Current Paper List"):
        gr.Markdown("# Current Paper List")
        with gr.Row():
            paper_list = gr.Textbox(label="Paper List")
        with gr.Row():
            reset_paper_list_button = gr.Button("Reset Paper List")
        reset_paper_list_button.click(reset_paper_list, inputs=[], outputs=[paper_list])
        
    with gr.Tab("Current Research Goals"):
        gr.Markdown("# Current Research Goals")
        with gr.Row():
            research_goals = gr.Textbox(label="Current Research Goals")
        with gr.Row():
            new_research_goals = gr.Textbox(label="Define new research goals")
        with gr.Row():
            research_button = gr.Button("Reset Research Goals")
        research_button.click(update_research_goals, inputs=[new_research_goals], outputs=[research_goals])
    app_demo.load(load_data_files, outputs=[research_goals, concept_base, paper_list])                  


app_demo.launch()