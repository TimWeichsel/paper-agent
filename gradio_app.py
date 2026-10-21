import gradio as gr
from matplotlib.figure import Figure
from src.agent.graph import graph
from src.services.db_client import get_all_concepts_from_db, get_all_concept_names_from_db, get_all_paper_titles_from_db, drop_db, drop_concept_table, initialize_db, get_concepts_with_embeddings, get_research_goal_from_db, update_research_goals_in_db
from src.plotting.plot_pca import update_pca_plot

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("src").setLevel(logging.DEBUG)

# Initialize the database when app starts
initialize_db()

def generate_new_pca_plot(indicate_last_concept: bool = False) -> Figure|None:
    '''Builds a PCA plot of the learned concepts and returns the matplotlib figure object'''
    concepts = get_concepts_with_embeddings()
    return update_pca_plot(concepts = concepts, indicate_last_concept = indicate_last_concept)

def learn_new_paper(learning_preferences: str = "", complexity_preference: str = "basic", learning_mode: str = "new"):
    '''Runs the agent graph while streaming the current status. At the end it shows whether a new concept was added to the database.'''
    agent_state = {"messages": [], "query_preference": learning_preferences, "complexity_preferences": complexity_preference, "learning_mode": learning_mode, "target_concept": None}
    result = {}
    try:
        for graph_update in graph.stream(agent_state, stream_mode="updates"):
            for node_update in graph_update.values():
                result.update(node_update or {})
            current_node_status = next(iter(graph_update))
            yield f"running: {current_node_status}", "", "", "", "", gr.skip(), gr.skip(), gr.skip(), gr.skip()
        _,concept_base,concept_names,paper_list = load_data()

        if result.get("saved_to_db"):
            final_status = "Saved to database"
        elif result.get("duplicate_of_concept"):
            final_status = f"Not saved to database as concept is a duplicate of: {result.get('duplicate_of_concept')}"
        else:
            final_status = "Not saved to database as Agent Validation failed or Network problem"

        if not result.get("saved_to_db"):
            result = {} #empty fields (no paper should be shown that was not saved to db)
        yield final_status, result.get("paper_title", ""), result.get("paper_concept", ""), result.get("paper_concept_definition", ""), result.get("concept_summary", ""), concept_base, concept_names, paper_list, generate_new_pca_plot(True)
    except Exception as e:
        logging.exception(f"Agent run failed: {str(e)}")
        yield f"Error: {str(e)}", "", "", "", "", gr.skip(), gr.skip(), gr.skip(), gr.skip()
        return

def load_data() -> tuple[str, str, str, str]:
    '''Loads and return the current research goals, concept base, concept names and paper titles from the database.'''
    return (
        get_research_goal_from_db(),
        "\n".join(get_all_concepts_from_db()),
        "\n".join(get_all_concept_names_from_db()),
        "\n".join(get_all_paper_titles_from_db())
    )

def reset_db():
    '''Drops all tables in databases, except the research goal entry'''
    drop_db()
    reset_concept_base()
    initialize_db()
    return ""

def reset_concept_base():
    '''Drops the concepts (not paper or authors) from the table.'''
    drop_concept_table()
    initialize_db()
    return ""

def update_research_goals(new_research_goals: str):
    '''Updates users' research goal in the database.'''
    update_research_goals_in_db(new_research_goals)
    return new_research_goals

with gr.Blocks() as app_demo:
    
    with gr.Tab("Learn Concept"):
        gr.Markdown("# Paper Learning Agent")
        with gr.Row():
            with gr.Column(scale=3):
                learning_input = gr.Textbox(label="Do you want to learn about anything specific next?")
                complexity_input = gr.Radio(choices=["basic","medium","advanced"],value="basic",label="Complexity Preference")
                learning_mode = gr.Radio(choices=["new", "related"],value="new",label="Learning Mode")
                status = gr.Textbox(label="Status", max_lines=1, value="Ready to Start", interactive=False)
                concept_button = gr.Button("Learn another concept")
            with gr.Column(scale=1):
                gr.Image(value="static_images/graph.png", label="Agent Graph")
        with gr.Row():
            paper_concept = gr.Textbox(label="Concept Name")
        with gr.Row():
            concept_definition = gr.Textbox(label="Concept Definition")
        with gr.Row():
            concept_summary = gr.Textbox(label="Concept Summary", max_lines=20)
        with gr.Row():
            paper_title = gr.Textbox(label="Paper Title")
        with gr.Row():
            with gr.Column(scale=2):
                concept_map = gr.Plot(label="Concept Map")
            with gr.Column(scale=1):
                concept_names = gr.Textbox(label="All Concepts Learned So Far", max_lines=20)

        
    
    with gr.Tab("Current Concept Base"):
        gr.Markdown("# Current Concept Base")
        with gr.Row():
            concept_base = gr.TextArea(label="Concepts Learned So Far", lines=50, max_lines=300, autoscroll=False)
        with gr.Row():
            reset_concept_base_button = gr.Button("Reset Concept Base")
        reset_concept_base_button.click(reset_concept_base, inputs=[], outputs=[concept_base])

    with gr.Tab("Current Paper List"):
        gr.Markdown("# Current Paper List")
        with gr.Row():
            paper_list = gr.TextArea(label="Paper List", lines=50, max_lines=300, autoscroll=False)
        with gr.Row():
            reset_db_button = gr.Button("Reset Paper and Concept Base")
        reset_db_button.click(reset_db, inputs=[], outputs=[paper_list])
        
        
    with gr.Tab("Current Research Goals"):
        gr.Markdown("# Current Research Goals")
        with gr.Row():
            research_goals = gr.TextArea(label="Current Research Goals", lines=30, max_lines=100, autoscroll=False)
        with gr.Row():
            new_research_goals = gr.Textbox(label="Define new Research goals")
        with gr.Row():
            research_button = gr.Button("Update Research Goals")
        research_button.click(update_research_goals, inputs=[new_research_goals], outputs=[research_goals])
    app_demo.load(load_data, outputs=[research_goals, concept_base, concept_names, paper_list])
    app_demo.load(generate_new_pca_plot, outputs=[concept_map])
    concept_button.click(learn_new_paper, inputs=[learning_input, complexity_input, learning_mode], outputs=[status, paper_title, paper_concept, concept_definition, concept_summary, concept_base, concept_names, paper_list, concept_map])              


app_demo.launch(server_name="0.0.0.0", theme=gr.themes.Ocean())