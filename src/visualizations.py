# src/visualizations.py - Reusable Visualization Components

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Dict


class EnergyVisualizations:
    """Reusable visualization components for energy data."""
    
    @staticmethod
    def create_sankey_diagram(sources: List[str], targets: List[str],
                             values: List[float], title: str = "Energy Flow") -> go.Figure:
        """
        Create Sankey diagram for energy flows.
        
        Args:
            sources: List of source countries
            targets: List of target countries
            values: Trade volumes
            title: Diagram title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color='black', width=0.5),
                label=list(set(sources + targets))
            ),
            link=dict(
                source=[list(set(sources + targets)).index(s) for s in sources],
                target=[list(set(sources + targets)).index(t) for t in targets],
                value=values
            )
        )])
        
        fig.update_layout(
            title_text=title,
            font_size=10,
            height=600,
            hovermode='closest'
        )
        
        return fig
    
    @staticmethod
    def create_treemap(labels: List[str], parents: List[str],
                      values: List[float], title: str = "Energy Distribution") -> go.Figure:
        """Create treemap for hierarchical energy data."""
        fig = go.Figure(go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(colorscale='RdYlGn', cmid=np.median(values))
        ))
        
        fig.update_layout(title_text=title, height=600)
        return fig
    
    @staticmethod
    def create_heatmap(data: pd.DataFrame, x_col: str, y_col: str,
                      value_col: str, title: str = "Heatmap") -> go.Figure:
        """Create heatmap for correlation analysis."""
        pivot_data = data.pivot_table(
            values=value_col,
            index=y_col,
            columns=x_col
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title_text=title,
            xaxis_title=x_col,
            yaxis_title=y_col,
            height=600
        )
        
        return fig
    
    @staticmethod
    def create_gauge(value: float, max_value: float = 100,
                    title: str = "Performance", unit: str = "%") -> go.Figure:
        """Create gauge chart for KPI visualization."""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            title={'text': title},
            gauge={
                'axis': {'range': [0, max_value]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, max_value*0.33], 'color': "lightgray"},
                    {'range': [max_value*0.33, max_value*0.67], 'color': "gray"},
                    {'range': [max_value*0.67, max_value], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_value*0.8
                }
            }
        ))
        
        fig.update_layout(height=400)
        return fig
    
    @staticmethod
    def create_waterfall(categories: List[str], values: List[float],
                        title: str = "Growth Waterfall") -> go.Figure:
        """Create waterfall chart for breakdown analysis."""
        fig = go.Figure(go.Waterfall(
            x=categories,
            y=values,
            connector={"line": {"color": "rgba(63, 63, 63, 0.5)"}},
            decreasing={"marker": {"color": "red"}},
            increasing={"marker": {"color": "green"}},
            totals={"marker": {"color": "blue"}}
        ))
        
        fig.update_layout(
            title_text=title,
            showlegend=False,
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_sunburst(labels: List[str], parents: List[str],
                       values: List[float], title: str = "Energy Hierarchy") -> go.Figure:
        """Create sunburst chart for hierarchical data."""
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(colorscale='RdBu')
        ))
        
        fig.update_layout(
            title_text=title,
            height=600
        )
        
        return fig
    
    @staticmethod
    def create_box_plot(data: pd.DataFrame, x_col: str, y_col: str,
                       title: str = "Distribution") -> go.Figure:
        """Create box plot for distribution analysis."""
        fig = px.box(
            data,
            x=x_col,
            y=y_col,
            title=title,
            points="outliers"
        )
        
        fig.update_layout(height=500)
        return fig
    
    @staticmethod
    def create_violin_plot(data: pd.DataFrame, x_col: str, y_col: str,
                          title: str = "Distribution") -> go.Figure:
        """Create violin plot for detailed distribution."""
        fig = px.violin(
            data,
            x=x_col,
            y=y_col,
            title=title,
            box=True,
            points="all"
        )
        
        fig.update_layout(height=500)
        return fig
    
    @staticmethod
    def create_parallel_categories(data: pd.DataFrame, dimensions: List[str],
                                  title: str = "Parallel Categories") -> go.Figure:
        """Create parallel categories plot."""
        fig = px.parallel_categories(
            data,
            dimensions=dimensions,
            title=title
        )
        
        fig.update_layout(height=600)
        return fig
    
    @staticmethod
    def create_3d_scatter(data: pd.DataFrame, x_col: str, y_col: str,
                         z_col: str, color_col: str = None,
                         title: str = "3D Analysis") -> go.Figure:
        """Create 3D scatter plot."""
        fig = px.scatter_3d(
            data,
            x=x_col,
            y=y_col,
            z=z_col,
            color=color_col,
            title=title,
            hover_data=data.columns
        )
        
        fig.update_layout(height=700)
        return fig
    
    @staticmethod
    def create_timeline(data: pd.DataFrame, date_col: str, y_col: str,
                       title: str = "Timeline") -> go.Figure:
        """Create timeline visualization."""
        fig = px.line(
            data,
            x=date_col,
            y=y_col,
            title=title,
            markers=True,
            template="plotly_white"
        )
        
        fig.update_xaxes(rangeslider_visible=True)
        fig.update_layout(height=500, hovermode='x unified')
        
        return fig
    
    @staticmethod
    def create_funnel(stages: List[str], values: List[float],
                     title: str = "Process Funnel") -> go.Figure:
        """Create funnel chart."""
        fig = go.Figure(go.Funnel(
            y=stages,
            x=values,
            textposition="inside",
            textinfo="value+percent initial"
        ))
        
        fig.update_layout(
            title_text=title,
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_scatter_matrix(data: pd.DataFrame, dimensions: List[str],
                             title: str = "Scatter Matrix") -> go.Figure:
        """Create scatter matrix for multi-variable analysis."""
        fig = px.scatter_matrix(
            data,
            dimensions=dimensions,
            title=title,
            height=800
        )
        
        return fig
    
    @staticmethod
    def create_strip_chart(data: pd.DataFrame, x_col: str, y_col: str,
                          color_col: str = None,
                          title: str = "Strip Chart") -> go.Figure:
        """Create strip chart for categorical data."""
        fig = px.strip(
            data,
            x=x_col,
            y=y_col,
            color=color_col,
            title=title,
            jitter=True
        )
        
        fig.update_layout(height=500)
        return fig


class GridVisualizationUtils:
    """Utility functions for grid-related visualizations."""
    
    @staticmethod
    def create_power_flow_diagram(nodes: List[Dict], edges: List[Dict]) -> go.Figure:
        """Create power flow network diagram."""
        # Extract positions and create edges
        x_nodes = [n['x'] for n in nodes]
        y_nodes = [n['y'] for n in nodes]
        node_labels = [n['label'] for n in nodes]
        
        edge_trace_x = []
        edge_trace_y = []
        
        for edge in edges:
            start_idx = edge['from']
            end_idx = edge['to']
            edge_trace_x.extend([nodes[start_idx]['x'], nodes[end_idx]['x'], None])
            edge_trace_y.extend([nodes[start_idx]['y'], nodes[end_idx]['y'], None])
        
        edge_trace = go.Scatter(
            x=edge_trace_x,
            y=edge_trace_y,
            mode='lines',
            line=dict(width=2, color='rgba(125,125,125,0.5)'),
            hoverinfo='none',
            showlegend=False
        )
        
        node_trace = go.Scatter(
            x=x_nodes,
            y=y_nodes,
            mode='markers+text',
            text=node_labels,
            textposition="top center",
            hoverinfo='label',
            marker=dict(
                showscale=True,
                size=20,
                color='lightblue',
                line_width=2
            ),
            showlegend=False
        )
        
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title='Power Grid Network',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            height=600
        )
        
        return fig
    
    @staticmethod
    def create_capacity_utilization_gauge(current: float, capacity: float,
                                         title: str = "Grid Utilization") -> go.Figure:
        """Create gauge for capacity utilization."""
        utilization = (current / capacity) * 100
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=utilization,
            title={'text': title},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}
                ]
            }
        ))
        
        fig.update_layout(height=400)
        return fig
