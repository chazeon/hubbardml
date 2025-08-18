"""Template-based output formatting for HubbardML predictions."""

import jinja2
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


def get_template_dir() -> Path:
    """Get the templates directory."""
    return Path(__file__).parent.parent.parent / "templates"


def setup_jinja_env() -> jinja2.Environment:
    """Setup Jinja2 environment with custom functions."""
    template_dir = get_template_dir()
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True
    )
    
    # Add custom functions for QE complexity
    env.globals['get_atom_index'] = get_atom_index
    env.globals['get_interaction_type'] = get_interaction_type
    env.globals['format_orbital'] = format_orbital
    
    return env


def get_atom_index(site_data: Dict) -> int:
    """Map site data to QE atom index."""
    # This is where complex QE neighbor mapping logic would go
    # For now, return a placeholder - users can customize this
    return site_data.get('qe_atom_index', 1)


def get_interaction_type(v_prediction: Dict) -> int:
    """Determine interaction type k (1-4) for QE V parameters."""
    # k=1: standard-standard (most common)
    # k=2: standard-background  
    # k=3: background-background
    # k=4: background-standard
    return 1  # Default to standard-standard


def format_orbital(element: str, orbital: str) -> str:
    """Format element-orbital pair consistently."""
    return f"{element}-{orbital}"


def prepare_template_data(
    u_predictions: List[Dict] = None,
    v_predictions: List[Dict] = None,
    model_name: str = "hubbardml_model",
    model_type: str = "mixed",
    system_name: str = "system",
    projection_type: str = "ortho-atomic",
    lda_plus_u_kind: int = 2,
    **kwargs
) -> Dict[str, Any]:
    """Prepare data for template rendering."""
    
    return {
        'model_name': model_name,
        'model_type': model_type,
        'timestamp': datetime.now().isoformat(),
        'system_name': system_name,
        'projection_type': projection_type,
        'lda_plus_u_kind': lda_plus_u_kind,
        'u_predictions': u_predictions or [],
        'v_predictions': v_predictions or [],
        **kwargs  # Allow additional custom data
    }


def render_template(
    template_name: str,
    template_data: Dict[str, Any],
    output_path: Optional[str] = None
) -> str:
    """Render template with data."""
    
    env = setup_jinja_env()
    
    # Handle both .j2 extension and without
    if not template_name.endswith('.j2'):
        template_name = f"{template_name}.j2"
    
    try:
        template = env.get_template(template_name)
        rendered = template.render(**template_data)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(rendered)
            print(f"Output written to: {output_path}")
        
        return rendered
        
    except jinja2.TemplateNotFound:
        available_templates = list(get_template_dir().glob("*.j2"))
        template_names = [t.stem for t in available_templates]
        raise ValueError(
            f"Template '{template_name}' not found. "
            f"Available templates: {template_names}"
        )


def render_predictions(
    u_predictions: List[Dict] = None,
    v_predictions: List[Dict] = None,
    template: str = "qe_simple",
    output_path: Optional[str] = None,
    **template_params
) -> str:
    """Render predictions using Jinja2 templates."""
    
    template_data = prepare_template_data(
        u_predictions=u_predictions,
        v_predictions=v_predictions,
        **template_params
    )
    
    return render_template(template, template_data, output_path)


# Example usage and data structure documentation

def render_predictions_example():
    """Example usage of the unified template system."""
    
    u_predictions = [{
        'element': 'Co',
        'orbital': '3d', 
        'value': 7.70,
        'confidence': 0.95,
        'site_data': {'qe_atom_index': 1}
    }]
    
    v_predictions = [{
        'site0_element': 'Co',
        'site0_orbital': '3d',
        'site1_element': 'O', 
        'site1_orbital': '2p',
        'value': 0.75,
        'confidence': 0.92,
        'site0_data': {'qe_atom_index': 1},
        'site1_data': {'qe_atom_index': 19}
    }]
    
    # QE format
    qe_output = render_predictions(u_predictions, v_predictions, template="qe_simple")
    
    # JSON format
    json_output = render_predictions(u_predictions, v_predictions, template="json")
    
    # Custom template variables
    custom_output = render_predictions(
        u_predictions, v_predictions, 
        template="qe_simple",
        projection_type="atomic",
        lda_plus_u_kind=2
    )