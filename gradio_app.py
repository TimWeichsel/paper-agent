import gradio as gr
from utils.file_utils import load_file
from src.agent.graph import graph
import uuid

def learn_new_paper(learning_preferences: str = "", complexity_preference: str = "medium", learning_mode: str = "new") -> str:
    id = uuid.uuid4()
    config = {"configurable": {"thread_id": str(id)}}
    agent_state = {"messages": [], "query_preference": learning_preferences, "complexity_preferences": complexity_preference, "learning_mode": learning_mode}
    result = graph.invoke(agent_state, config=config)
    concept_base = load_file("src/data/concept_base.txt", "No concepts learned yet")
    concept_names = load_file("src/data/concept_names.txt", "No concepts learned yet")
    paper_list = load_file("src/data/paper_list.txt", "No papers analyzed yet")
    return result["paper_title"], result["paper_concept"], result["paper_summary"], concept_base, concept_names, paper_list

def load_data_files():
    return (
        load_file("src/data/research_goals.txt", "No knowledge base found"),
        load_file("src/data/concept_base.txt", "No concepts analyzed yet"),
        load_file("src/data/concept_names.txt", "No concepts analyzed yet"),
        load_file("src/data/paper_list.txt", "No papers analyzed yet")
    )

def reset_paper_list():
    from utils.file_utils import save_file
    save_file("src/data/paper_list.txt", "")
    return ""

def reset_concept_base():
    from utils.file_utils import save_file
    save_file("src/data/concept_base.txt", "")
    return ""

def update_research_goals(new_research_goals: str):
    from utils.file_utils import save_file
    save_file("src/data/research_goals.txt", new_research_goals)
    return new_research_goals

with gr.Blocks() as app_demo:
    
    with gr.Tab("Learn Concept"):
        gr.Markdown("# Paper Learning Agent")
        with gr.Row():
            learning_input = gr.Textbox(label="Do you want to learn about anything specific next?")
        with gr.Row():
            learning_mode = gr.Radio(choices=["new", "related"],value="new",label="Learning Mode")
        with gr.Row():
            complexity_input = gr.Radio(choices=["basic","medium","advanced"],value="basic",label="Complexity Preference")
        with gr.Row():
            concept_button = gr.Button("Learn another concept")
        with gr.Row():
            paper_title = gr.Textbox(label="Paper Title")
        with gr.Row():
            paper_concept = gr.Textbox(label="Learned Concept")
        with gr.Row():
            paper_summary = gr.Textbox(label="Paper Summary", max_lines=20)
        with gr.Row():
            concept_names = gr.Textbox(label="All Concepts Learned So Far", max_lines=20)
        
    
    with gr.Tab("Current Concept Base"):
        gr.Markdown("# Current Concept Base")
        with gr.Row():
            concept_base = gr.Textbox(label="Concepts Learned So Far", max_lines=50)
        with gr.Row():
            reset_concept_base_button = gr.Button("Reset Concept Base")
        reset_concept_base_button.click(reset_concept_base, inputs=[], outputs=[concept_base])

    with gr.Tab("Current Paper List"):
        gr.Markdown("# Current Paper List")
        with gr.Row():
            paper_list = gr.Textbox(label="Paper List", max_lines=50)
        with gr.Row():
            reset_paper_list_button = gr.Button("Reset Paper List")
        reset_paper_list_button.click(reset_paper_list, inputs=[], outputs=[paper_list])
        
    with gr.Tab("Current Research Goals"):
        gr.Markdown("# Current Research Goals")
        with gr.Row():
            research_goals = gr.Textbox(label="Current Research Goals")
        with gr.Row():
            new_research_goals = gr.Textbox(label="Define new Research goals")
        with gr.Row():
            research_button = gr.Button("Reset Research Goals")
        research_button.click(update_research_goals, inputs=[new_research_goals], outputs=[research_goals])
    app_demo.load(load_data_files, outputs=[research_goals, concept_base, concept_names, paper_list])    
    concept_button.click(learn_new_paper, inputs=[learning_input, complexity_input, learning_mode], outputs=[paper_title, paper_concept, paper_summary, concept_base, concept_names, paper_list])              


app_demo.launch()