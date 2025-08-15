"""
Figma Integration Tool  
Embedded Figma API integration for AI-driven design

This module provides:
- Design system creation and management
- Component library management  
- Asset pipeline automation
- Design handoff workflows
- Prototype creation and testing
- Design version control
- Team collaboration features
"""

import asyncio
import json
import requests
import base64
import io
import time
import os
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class FigmaNodeType(Enum):
    """Figma node types"""
    CANVAS = "CANVAS"
    FRAME = "FRAME"
    GROUP = "GROUP"
    VECTOR = "VECTOR"
    BOOLEAN_OPERATION = "BOOLEAN_OPERATION"
    STAR = "STAR"
    LINE = "LINE"
    ELLIPSE = "ELLIPSE"
    REGULAR_POLYGON = "REGULAR_POLYGON"
    RECTANGLE = "RECTANGLE"
    TEXT = "TEXT"
    SLICE = "SLICE"
    COMPONENT = "COMPONENT"
    COMPONENT_SET = "COMPONENT_SET"
    INSTANCE = "INSTANCE"

class FigmaExportFormat(Enum):
    """Figma export formats"""
    PNG = "PNG"
    JPG = "JPG" 
    SVG = "SVG"
    PDF = "PDF"

@dataclass
class FigmaColor:
    """Figma color representation"""
    r: float
    g: float
    b: float
    a: float = 1.0
    
    def to_hex(self) -> str:
        """Convert to hex color"""
        r = int(self.r * 255)
        g = int(self.g * 255) 
        b = int(self.b * 255)
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def to_rgba(self) -> str:
        """Convert to RGBA string"""
        return f"rgba({int(self.r*255)}, {int(self.g*255)}, {int(self.b*255)}, {self.a})"

@dataclass
class FigmaNode:
    """Figma design node"""
    id: str
    name: str
    type: FigmaNodeType
    visible: bool = True
    locked: bool = False
    children: List['FigmaNode'] = field(default_factory=list)
    fills: List[Dict] = field(default_factory=list)
    strokes: List[Dict] = field(default_factory=list)
    effects: List[Dict] = field(default_factory=list)
    
@dataclass
class FigmaComponent:
    """Figma component definition"""
    id: str
    name: str
    description: str
    node_id: str
    thumbnail_url: str = ""
    created_at: str = ""
    updated_at: str = ""

@dataclass
class FigmaProject:
    """Figma project structure"""
    id: str
    name: str
    files: List[Dict] = field(default_factory=list)
    created_at: str = ""
    modified_at: str = ""

class FigmaIntegration:
    """Figma API integration for AI-driven design"""
    
    def __init__(self, api_token: str = None, workspace_path: str = "/media/r/Workspace"):
        self.api_token = api_token
        self.workspace_path = workspace_path
        self.base_url = "https://api.figma.com/v1"
        self.is_connected = False
        self.projects = {}
        self.components = {}
        self.design_tokens = {}
        self.logger = logging.getLogger("figma_integration")
        
        # Initialize headers
        self.headers = {
            'X-Figma-Token': self.api_token if self.api_token else 'simulation_mode'
        }
        
    async def initialize(self) -> bool:
        """Initialize Figma integration"""
        try:
            if self.api_token:
                # Test API connection
                response = await self._make_request('GET', '/me')
                if response and response.get('id'):
                    self.is_connected = True
                    self.logger.info("Figma integration initialized successfully")
                    await self._load_projects()
                    return True
            else:
                self.logger.info("Figma running in simulation mode (no API token)")
                self.is_connected = False
                await self._setup_simulation_mode()
                return True
                
        except Exception as e:
            self.logger.error(f"Figma initialization failed: {e}")
            return False
    
    async def _setup_simulation_mode(self):
        """Setup simulation mode with mock data"""
        self.projects = {
            'sim_project_1': FigmaProject(
                id='sim_project_1',
                name='AI Generated Design System',
                files=[
                    {'key': 'sim_file_1', 'name': 'Components Library'},
                    {'key': 'sim_file_2', 'name': 'Design Tokens'},
                ]
            )
        }
        
        self.components = {
            'button_primary': FigmaComponent(
                id='comp_1',
                name='Primary Button',
                description='Main call-to-action button',
                node_id='1:1'
            ),
            'card_default': FigmaComponent(
                id='comp_2', 
                name='Default Card',
                description='Standard content card',
                node_id='1:2'
            )
        }
        
        self.design_tokens = {
            'colors': {
                'primary': FigmaColor(0.2, 0.4, 0.8, 1.0),
                'secondary': FigmaColor(0.5, 0.5, 0.5, 1.0),
                'success': FigmaColor(0.2, 0.7, 0.3, 1.0),
                'warning': FigmaColor(1.0, 0.8, 0.2, 1.0),
                'error': FigmaColor(0.9, 0.2, 0.2, 1.0)
            },
            'typography': {
                'heading': {'fontSize': 24, 'fontWeight': 'bold'},
                'body': {'fontSize': 16, 'fontWeight': 'normal'},
                'caption': {'fontSize': 12, 'fontWeight': 'normal'}
            },
            'spacing': {
                'xs': 4, 's': 8, 'm': 16, 'l': 24, 'xl': 32
            }
        }
        
        self.logger.info("Simulation mode setup complete")
    
    async def _make_request(self, method: str, endpoint: str, data: dict = None) -> Optional[Dict]:
        """Make API request to Figma"""
        if not self.is_connected:
            return self._simulate_api_response(endpoint)
            
        try:
            url = f"{self.base_url}{endpoint}"
            
            if method == 'GET':
                response = requests.get(url, headers=self.headers)
            elif method == 'POST':
                response = requests.post(url, headers=self.headers, json=data)
            elif method == 'PUT':
                response = requests.put(url, headers=self.headers, json=data)
            else:
                return None
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"API request failed: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"API request error: {e}")
            return None
    
    def _simulate_api_response(self, endpoint: str) -> Dict:
        """Simulate API responses for testing"""
        if '/me' in endpoint:
            return {'id': 'sim_user_123', 'email': 'ai@simulation.com'}
        elif '/teams' in endpoint:
            return {'teams': [{'id': 'sim_team_1', 'name': 'AI Design Team'}]}
        elif '/projects' in endpoint:
            return {'projects': [p.__dict__ for p in self.projects.values()]}
        elif '/files' in endpoint:
            return {
                'document': {
                    'id': 'sim_file_1',
                    'name': 'Design System',
                    'children': []
                }
            }
        else:
            return {'status': 'simulated', 'endpoint': endpoint}
    
    async def _load_projects(self):
        """Load user projects"""
        try:
            response = await self._make_request('GET', '/teams/me/projects')
            if response and 'projects' in response:
                for proj_data in response['projects']:
                    project = FigmaProject(
                        id=proj_data['id'],
                        name=proj_data['name']
                    )
                    self.projects[proj_data['id']] = project
                    
        except Exception as e:
            self.logger.error(f"Failed to load projects: {e}")
    
    async def create_design_system(self, name: str, brand_colors: Dict[str, str] = None) -> Dict[str, Any]:
        """Create a new design system"""
        try:
            # Generate design tokens
            if not brand_colors:
                brand_colors = {
                    'primary': '#3B82F6',
                    'secondary': '#6B7280', 
                    'success': '#10B981',
                    'warning': '#F59E0B',
                    'error': '#EF4444'
                }
            
            design_system = {
                'name': name,
                'id': f"ds_{int(time.time())}",
                'colors': {},
                'typography': {
                    'h1': {'fontSize': 32, 'fontWeight': 700, 'lineHeight': 1.2},
                    'h2': {'fontSize': 24, 'fontWeight': 600, 'lineHeight': 1.3},
                    'h3': {'fontSize': 20, 'fontWeight': 600, 'lineHeight': 1.4},
                    'body': {'fontSize': 16, 'fontWeight': 400, 'lineHeight': 1.5},
                    'caption': {'fontSize': 14, 'fontWeight': 400, 'lineHeight': 1.4}
                },
                'spacing': {
                    'xs': 4, 'sm': 8, 'md': 16, 'lg': 24, 'xl': 32, '2xl': 48
                },
                'borderRadius': {
                    'none': 0, 'sm': 4, 'md': 8, 'lg': 12, 'xl': 16, 'full': 9999
                },
                'shadows': {
                    'sm': '0 1px 2px rgba(0,0,0,0.05)',
                    'md': '0 4px 6px rgba(0,0,0,0.1)', 
                    'lg': '0 10px 15px rgba(0,0,0,0.1)',
                    'xl': '0 20px 25px rgba(0,0,0,0.1)'
                }
            }
            
            # Convert hex colors to Figma color format
            for name, hex_color in brand_colors.items():
                r = int(hex_color[1:3], 16) / 255
                g = int(hex_color[3:5], 16) / 255
                b = int(hex_color[5:7], 16) / 255
                design_system['colors'][name] = FigmaColor(r, g, b, 1.0)
            
            self.design_tokens[design_system['id']] = design_system
            
            # Save design system to file
            await self._save_design_system(design_system)
            
            self.logger.info(f"Created design system: {name}")
            return {
                'success': True,
                'design_system_id': design_system['id'],
                'name': name,
                'tokens_count': len(design_system['colors']) + len(design_system['typography'])
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create design system: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _save_design_system(self, design_system: Dict):
        """Save design system to workspace"""
        try:
            output_dir = os.path.join(self.workspace_path, 'design_systems')
            os.makedirs(output_dir, exist_ok=True)
            
            # Save as JSON
            json_path = os.path.join(output_dir, f"{design_system['name'].lower().replace(' ', '_')}.json")
            
            # Convert FigmaColor objects to dict for JSON serialization
            serializable_ds = design_system.copy()
            serializable_ds['colors'] = {
                k: {'r': v.r, 'g': v.g, 'b': v.b, 'a': v.a, 'hex': v.to_hex()}
                for k, v in design_system['colors'].items()
            }
            
            with open(json_path, 'w') as f:
                json.dump(serializable_ds, f, indent=2)
            
            # Generate CSS variables
            css_path = os.path.join(output_dir, f"{design_system['name'].lower().replace(' ', '_')}.css")
            css_content = self._generate_css_variables(design_system)
            
            with open(css_path, 'w') as f:
                f.write(css_content)
            
            self.logger.info(f"Design system saved to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save design system: {e}")
    
    def _generate_css_variables(self, design_system: Dict) -> str:
        """Generate CSS custom properties from design system"""
        css = f"/* {design_system['name']} Design System */\\n:root {{\\n"
        
        # Colors
        css += "  /* Colors */\\n"
        for name, color in design_system['colors'].items():
            css += f"  --color-{name}: {color.to_hex()};\\n"
        
        # Typography
        css += "\\n  /* Typography */\\n"
        for name, props in design_system['typography'].items():
            css += f"  --font-size-{name}: {props['fontSize']}px;\\n"
            css += f"  --font-weight-{name}: {props['fontWeight']};\\n"
            css += f"  --line-height-{name}: {props['lineHeight']};\\n"
        
        # Spacing
        css += "\\n  /* Spacing */\\n"
        for name, value in design_system['spacing'].items():
            css += f"  --spacing-{name}: {value}px;\\n"
        
        # Border radius
        css += "\\n  /* Border Radius */\\n"
        for name, value in design_system['borderRadius'].items():
            css += f"  --radius-{name}: {value}px;\\n"
        
        css += "}\\n"
        return css
    
    async def create_component(self, name: str, description: str, properties: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a new component"""
        try:
            if not properties:
                properties = self._get_default_component_properties(name)
            
            component = FigmaComponent(
                id=f"comp_{int(time.time())}",
                name=name,
                description=description,
                node_id=f"node_{int(time.time())}"
            )
            
            self.components[component.id] = component
            
            # Generate component code
            component_code = await self._generate_component_code(component, properties)
            
            # Save component
            await self._save_component(component, component_code)
            
            self.logger.info(f"Created component: {name}")
            return {
                'success': True,
                'component_id': component.id,
                'name': name,
                'code_generated': bool(component_code)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create component: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_default_component_properties(self, name: str) -> Dict[str, Any]:
        """Get default properties for common components"""
        defaults = {
            'button': {
                'variant': ['primary', 'secondary', 'outline'],
                'size': ['sm', 'md', 'lg'],
                'disabled': False,
                'loading': False
            },
            'card': {
                'shadow': ['none', 'sm', 'md', 'lg'],
                'padding': ['sm', 'md', 'lg'],
                'border': True
            },
            'input': {
                'type': ['text', 'email', 'password', 'number'],
                'size': ['sm', 'md', 'lg'], 
                'disabled': False,
                'error': False
            },
            'modal': {
                'size': ['sm', 'md', 'lg', 'xl'],
                'closable': True,
                'backdrop': True
            }
        }
        
        component_type = name.lower().split('_')[0]
        return defaults.get(component_type, {'customizable': True})
    
    async def _generate_component_code(self, component: FigmaComponent, properties: Dict) -> str:
        """Generate React/TypeScript code for component"""
        try:
            # Generate interface for props
            props_interface = f"interface {component.name.replace(' ', '')}Props {{\\n"
            for prop, config in properties.items():
                if isinstance(config, list):
                    props_interface += f"  {prop}?: '{'\' | \''.join(config)}';\\n"
                elif isinstance(config, bool):
                    props_interface += f"  {prop}?: boolean;\\n"
                else:
                    props_interface += f"  {prop}?: any;\\n"
            props_interface += "}\\n\\n"
            
            # Generate component code
            component_name = component.name.replace(' ', '')
            code = f"""import React from 'react';
import './styles.css';

{props_interface}export const {component_name}: React.FC<{component_name}Props> = ({{
{', '.join(properties.keys())}
}}) => {{
  return (
    <div className="{component_name.lower()}">
      {{/* Component implementation */}}
      <span>{{children}}</span>
    </div>
  );
}};

export default {component_name};
"""
            
            return code
            
        except Exception as e:
            self.logger.error(f"Failed to generate component code: {e}")
            return ""
    
    async def _save_component(self, component: FigmaComponent, code: str):
        """Save component to workspace"""
        try:
            components_dir = os.path.join(self.workspace_path, 'ui', 'components')
            os.makedirs(components_dir, exist_ok=True)
            
            component_name = component.name.replace(' ', '')
            component_dir = os.path.join(components_dir, component_name)
            os.makedirs(component_dir, exist_ok=True)
            
            # Save TypeScript component
            ts_path = os.path.join(component_dir, f'{component_name}.tsx')
            with open(ts_path, 'w') as f:
                f.write(code)
            
            # Save component metadata
            metadata = {
                'id': component.id,
                'name': component.name,
                'description': component.description,
                'created_at': time.time(),
                'figma_node_id': component.node_id
            }
            
            metadata_path = os.path.join(component_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Component saved to {component_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save component: {e}")
    
    async def export_assets(self, file_key: str, node_ids: List[str], format: FigmaExportFormat = FigmaExportFormat.PNG) -> Dict[str, Any]:
        """Export assets from Figma"""
        try:
            if not self.is_connected:
                # Simulate export
                return {
                    'success': True,
                    'simulated': True,
                    'exported_count': len(node_ids),
                    'format': format.value
                }
            
            # Real API export
            params = {
                'ids': ','.join(node_ids),
                'format': format.value.lower()
            }
            
            response = await self._make_request('GET', f'/images/{file_key}', params)
            
            if response and 'images' in response:
                exported = []
                assets_dir = os.path.join(self.workspace_path, 'assets')
                os.makedirs(assets_dir, exist_ok=True)
                
                for node_id, image_url in response['images'].items():
                    if image_url:
                        # Download image
                        img_response = requests.get(image_url)
                        if img_response.status_code == 200:
                            filename = f"{node_id}.{format.value.lower()}"
                            filepath = os.path.join(assets_dir, filename)
                            
                            with open(filepath, 'wb') as f:
                                f.write(img_response.content)
                            
                            exported.append({
                                'node_id': node_id,
                                'filename': filename,
                                'path': filepath
                            })
                
                return {
                    'success': True,
                    'exported': exported,
                    'exported_count': len(exported)
                }
            
            return {'success': False, 'error': 'No images returned'}
            
        except Exception as e:
            self.logger.error(f"Failed to export assets: {e}")
            return {'success': False, 'error': str(e)}
    
    async def create_prototype(self, file_key: str, flows: List[Dict] = None) -> Dict[str, Any]:
        """Create interactive prototype"""
        try:
            if not flows:
                flows = [
                    {'from': 'home', 'to': 'details', 'trigger': 'click'},
                    {'from': 'details', 'to': 'home', 'trigger': 'back'}
                ]
            
            prototype = {
                'id': f"proto_{int(time.time())}",
                'file_key': file_key,
                'flows': flows,
                'created_at': time.time(),
                'status': 'ready'
            }
            
            # Generate prototype specification
            spec = await self._generate_prototype_spec(prototype)
            
            # Save prototype
            await self._save_prototype(prototype, spec)
            
            self.logger.info(f"Created prototype for file {file_key}")
            return {
                'success': True,
                'prototype_id': prototype['id'],
                'flows_count': len(flows),
                'spec_generated': bool(spec)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create prototype: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _generate_prototype_spec(self, prototype: Dict) -> str:
        """Generate prototype specification"""
        spec = f"""# Prototype Specification
ID: {prototype['id']}
File: {prototype['file_key']}
Created: {time.ctime(prototype['created_at'])}

## User Flows
"""
        
        for i, flow in enumerate(prototype['flows'], 1):
            spec += f"{i}. {flow['from']} → {flow['to']} (trigger: {flow['trigger']})\\n"
        
        spec += """
## Implementation Notes
- Use React Router for navigation
- Implement smooth transitions
- Add loading states
- Handle error states
- Optimize for mobile
"""
        
        return spec
    
    async def _save_prototype(self, prototype: Dict, spec: str):
        """Save prototype to workspace"""
        try:
            prototypes_dir = os.path.join(self.workspace_path, 'prototypes')
            os.makedirs(prototypes_dir, exist_ok=True)
            
            # Save prototype data
            proto_path = os.path.join(prototypes_dir, f"{prototype['id']}.json")
            with open(proto_path, 'w') as f:
                json.dump(prototype, f, indent=2)
            
            # Save specification
            spec_path = os.path.join(prototypes_dir, f"{prototype['id']}_spec.md")
            with open(spec_path, 'w') as f:
                f.write(spec)
            
            self.logger.info(f"Prototype saved to {prototypes_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save prototype: {e}")
    
    async def get_design_handoff(self, file_key: str) -> Dict[str, Any]:
        """Generate design handoff documentation"""
        try:
            handoff = {
                'file_key': file_key,
                'generated_at': time.time(),
                'design_tokens': self.design_tokens,
                'components': [comp.__dict__ for comp in self.components.values()],
                'assets': [],
                'specifications': {}
            }
            
            # Generate CSS output
            css_output = ""
            for ds_id, design_system in self.design_tokens.items():
                css_output += self._generate_css_variables(design_system)
            
            # Generate component documentation
            component_docs = await self._generate_component_docs()
            
            handoff['css_output'] = css_output
            handoff['component_docs'] = component_docs
            
            # Save handoff package
            await self._save_handoff_package(handoff)
            
            self.logger.info(f"Generated design handoff for {file_key}")
            return {
                'success': True,
                'components_count': len(handoff['components']),
                'tokens_count': len(handoff['design_tokens']),
                'css_generated': bool(css_output)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate design handoff: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _generate_component_docs(self) -> str:
        """Generate component documentation"""
        docs = "# Component Library\\n\\n"
        
        for comp in self.components.values():
            docs += f"## {comp.name}\\n"
            docs += f"{comp.description}\\n\\n"
            docs += f"**Component ID:** `{comp.id}`\\n"
            docs += f"**Figma Node:** `{comp.node_id}`\\n\\n"
            docs += "### Usage\\n```tsx\\n"
            docs += f"import {{ {comp.name.replace(' ', '')} }} from './components/{comp.name.replace(' ', '')}'\\n\\n"
            docs += f"<{comp.name.replace(' ', '')} />\\n"
            docs += "```\\n\\n"
        
        return docs
    
    async def _save_handoff_package(self, handoff: Dict):
        """Save design handoff package"""
        try:
            handoff_dir = os.path.join(self.workspace_path, 'design_handoff')
            os.makedirs(handoff_dir, exist_ok=True)
            
            # Save handoff data
            handoff_path = os.path.join(handoff_dir, 'handoff.json')
            
            # Make handoff JSON serializable
            serializable_handoff = handoff.copy()
            serializable_handoff['design_tokens'] = {
                ds_id: {
                    'name': ds['name'],
                    'colors': {k: v.to_hex() if hasattr(v, 'to_hex') else str(v) 
                              for k, v in ds.get('colors', {}).items()},
                    'typography': ds.get('typography', {}),
                    'spacing': ds.get('spacing', {})
                }
                for ds_id, ds in handoff['design_tokens'].items()
            }
            
            with open(handoff_path, 'w') as f:
                json.dump(serializable_handoff, f, indent=2)
            
            # Save CSS
            if handoff.get('css_output'):
                css_path = os.path.join(handoff_dir, 'design_tokens.css')
                with open(css_path, 'w') as f:
                    f.write(handoff['css_output'])
            
            # Save component docs
            if handoff.get('component_docs'):
                docs_path = os.path.join(handoff_dir, 'components.md')
                with open(docs_path, 'w') as f:
                    f.write(handoff['component_docs'])
            
            self.logger.info(f"Design handoff saved to {handoff_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save handoff package: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get Figma integration status"""
        return {
            'connected': self.is_connected,
            'api_token_configured': bool(self.api_token),
            'projects_count': len(self.projects),
            'components_count': len(self.components),
            'design_tokens_count': len(self.design_tokens),
            'simulation_mode': not self.is_connected,
            'capabilities': [
                'design_system_creation',
                'component_management',
                'asset_export',
                'prototype_creation',
                'design_handoff',
                'css_generation'
            ]
        }

# Global Figma instance
figma_tool = FigmaIntegration()

async def initialize_figma_tool(api_token: str = None) -> bool:
    """Initialize Figma tool"""
    if api_token:
        figma_tool.api_token = api_token
        figma_tool.headers['X-Figma-Token'] = api_token
    return await figma_tool.initialize()

if __name__ == "__main__":
    import asyncio
    
    async def demo():
        print("🎨 Figma Integration Tool Demo")
        print("=" * 50)
        
        success = await figma_tool.initialize()
        print(f"Initialization: {'✅ Success' if success else '❌ Failed'}")
        
        if success:
            # Demo design system creation
            ds_result = await figma_tool.create_design_system(
                "AI Design System",
                {
                    'primary': '#3B82F6',
                    'secondary': '#6B7280',
                    'success': '#10B981'
                }
            )
            print(f"Design System: {'✅ Created' if ds_result['success'] else '❌ Failed'}")
            
            # Demo component creation
            comp_result = await figma_tool.create_component(
                "Primary Button",
                "Main call-to-action button component"
            )
            print(f"Component: {'✅ Created' if comp_result['success'] else '❌ Failed'}")
            
            # Demo status
            status = figma_tool.get_status()
            print(f"Status: {status}")
        
        print("Demo complete!")
    
    asyncio.run(demo())
