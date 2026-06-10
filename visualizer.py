import networkx as nx
import matplotlib.pyplot as plt


class NeuralVisualizer:

    def __init__(self, output_path="training_plot.png", smoothing_alpha=0.2, batches_per_epoch=1793):

        self.output_path = output_path
        self.G = nx.DiGraph()

        # store batches per epoch for global batch calculation
        self.batches_per_epoch = batches_per_epoch

        # nodes
        self.input_nodes = ["Input"]
        self.gru_nodes = [f"G{i}" for i in range(8)]
        self.delta_nodes = [f"D{i}" for i in range(4)]
        self.output_nodes = ["Output"]

        for node in (
            self.input_nodes
            + self.gru_nodes
            + self.delta_nodes
            + self.output_nodes
        ):
            self.G.add_node(node)

        # edges: input -> all 12 middle nodes -> output
        for inp in self.input_nodes:
            for gru in self.gru_nodes:
                self.G.add_edge(inp, gru)
            for delta in self.delta_nodes:
                self.G.add_edge(inp, delta)

        for gru in self.gru_nodes:
            self.G.add_edge(gru, "Output")
        for delta in self.delta_nodes:
            self.G.add_edge(delta, "Output")

        # positions: horizontal layout
        self.pos = {}
        # input at top center
        total = len(self.gru_nodes) + len(self.delta_nodes)
        mid_x = (total - 1) / 2.0
        self.pos["Input"] = (mid_x, 2.0)

        # middle row: 8 GRU then 4 delta
        middle_y = 1.0
        for i, n in enumerate(self.gru_nodes + self.delta_nodes):
            self.pos[n] = (i * 1.0, middle_y)

        # output at bottom
        self.pos["Output"] = (mid_x, 0.0)

        # smoothing (EMA)
        self.smoothing_alpha = smoothing_alpha
        self.prev_gru = None
        self.prev_delta = None
        self.prev_input = None

        # track MAE history for plotting
        self.mae_history = []
        self.batch_history = []

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 4))
        self.fig.show()
        self.fig.canvas.draw()

    def show(self):
        self.fig.show()
        self.fig.canvas.draw()
        self.fig.savefig(self.output_path)
        plt.pause(0.01)

    def plot_mae_graph(self):
        """Plot and save MAE vs batch number to mae_graph.png"""
        if len(self.mae_history) == 0:
            return

        fig_mae, ax_mae = plt.subplots(figsize=(10, 5))
        ax_mae.plot(self.batch_history, self.mae_history, linewidth=1.5, color='#FF6B6B', marker='o', markersize=3)
        ax_mae.set_xlabel('Batch Number', fontsize=12)
        ax_mae.set_ylabel('Percent Error', fontsize=12)
        ax_mae.set_title('Training Percent Error over Batches', fontsize=14, weight='bold')
        ax_mae.grid(True, alpha=0.3)
        fig_mae.tight_layout()
        fig_mae.savefig('mae_graph.png', dpi=100)
        plt.close(fig_mae)


    def update(self, epoch, batch, mae, input_val, gru_values, delta_values):
        self.ax.clear()

        # track MAE history (global batch counter)
        global_batch = (epoch - 1) * self.batches_per_epoch + batch
        self.mae_history.append(mae)
        self.batch_history.append(global_batch)

        # get magnitudes and apply EMA smoothing
        gru_strength = abs(gru_values[: len(self.gru_nodes)])
        delta_strength = abs(delta_values[: len(self.delta_nodes)])

        if self.prev_gru is None:
            self.prev_gru = gru_strength.copy()
        else:
            self.prev_gru = self.smoothing_alpha * gru_strength + (1 - self.smoothing_alpha) * self.prev_gru

        if self.prev_delta is None:
            self.prev_delta = delta_strength.copy()
        else:
            self.prev_delta = self.smoothing_alpha * delta_strength + (1 - self.smoothing_alpha) * self.prev_delta

        if self.prev_input is None:
            self.prev_input = float(input_val)
        else:
            self.prev_input = self.smoothing_alpha * float(input_val) + (1 - self.smoothing_alpha) * self.prev_input

        gru_strength = self.prev_gru
        delta_strength = self.prev_delta
        input_val = self.prev_input

        # normalization
        max_val = 1e-6
        if len(gru_strength) > 0:
            max_val = max(max_val, float(gru_strength.max()))
        if len(delta_strength) > 0:
            max_val = max(max_val, float(delta_strength.max()))
        max_val = max(max_val, abs(float(input_val)))

        # helpers for color blending
        def hex_to_rgb(h):
            h = h.lstrip('#')
            return tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))

        def rgb_to_hex(rgb):
            return '#%02x%02x%02x' % tuple(int(max(0, min(255, round(c*255)))) for c in rgb)

        def blend_with_white(base_hex, factor):
            # factor 0..1 -> amount of base color (small factor keeps it light)
            br, bg, bb = hex_to_rgb(base_hex)
            wr, wg, wb = (1.0, 1.0, 1.0)
            r = wr * (1-factor) + br * factor
            g = wg * (1-factor) + bg * factor
            b = wb * (1-factor) + bb * factor
            return rgb_to_hex((r, g, b))

        base_gru = '#FFF59D'   # pastel yellow
        base_delta = '#FFD3A5' # pastel orange
        base_input = '#E3F2FD' # light blue

        node_colors = []
        labels = {}

        for node in self.G.nodes():
            if node == 'Input':
                labels[node] = 'Input'
                val = float(input_val)
                factor = min(1.0, abs(val) / max_val) * 0.25
                node_colors.append(blend_with_white(base_input, factor))

            elif node.startswith('G'):
                idx = int(node[1:])
                if idx < len(gru_strength):
                    val = float(gru_strength[idx])
                    labels[node] = f"{val:.3f}"
                    factor = min(1.0, abs(val) / max_val) * 0.5
                    node_colors.append(blend_with_white(base_gru, factor))
                else:
                    labels[node] = '0.000'
                    node_colors.append(blend_with_white(base_gru, 0.0))

            elif node.startswith('D'):
                idx = int(node[1:])
                if idx < len(delta_strength):
                    val = float(delta_strength[idx])
                    labels[node] = f"{val:.3f}"
                    factor = min(1.0, abs(val) / max_val) * 0.6
                    node_colors.append(blend_with_white(base_delta, factor))
                else:
                    labels[node] = '0.000'
                    node_colors.append(blend_with_white(base_delta, 0.1))

            else:
                # skip output from networkx draw; we'll draw it manually as a rectangle
                continue

        # draw only input + 12 middle nodes (exclude output)
        visible_nodes = [n for n in self.G.nodes() if n != 'Output']
        visible_edges = [(u, v) for u, v in self.G.edges() if u != 'Output' and v != 'Output']

        # add edges from all 12 middle nodes to Output
        for node in self.gru_nodes + self.delta_nodes:
            visible_edges.append((node, 'Output'))

        nx.draw_networkx_nodes(
            self.G,
            self.pos,
            nodelist=visible_nodes,
            node_color=node_colors,
            node_size=1200,
            ax=self.ax
        )

        nx.draw_networkx_labels(self.G, self.pos, labels, ax=self.ax)

        nx.draw_networkx_edges(
            self.G,
            self.pos,
            edgelist=visible_edges,
            edge_color='gray',
            width=0.5,
            ax=self.ax
        )

        # Draw output as a text box with black border and white fill
        from matplotlib.patches import FancyBboxPatch
        output_x, output_y = self.pos['Output']
        percent_error = mae
        rect = FancyBboxPatch((output_x - 1.0, output_y - 0.2), 2.0, 0.4,
                              boxstyle="round,pad=0.02", edgecolor='black', facecolor='white',
                              linewidth=1.5)
        self.ax.add_patch(rect)
        self.ax.text(output_x, output_y, f"Percent Error: {percent_error:.1f}%", ha='center', va='center',
                    fontsize=9, weight='bold')

        # Draw boxes around sequence (8 GRU) and delta (4) groups
        from matplotlib.patches import Rectangle

        # sequence box (8 GRU nodes on left, indices 0-7)
        seq_x_min, seq_x_max = -0.5, 7.5
        seq_y_min, seq_y_max = 0.6, 1.4
        seq_rect = Rectangle((seq_x_min, seq_y_min), seq_x_max - seq_x_min, seq_y_max - seq_y_min,
                             linewidth=2, edgecolor='#FFC107', facecolor='none', linestyle='--')
        self.ax.add_patch(seq_rect)
        self.ax.text(seq_x_min - 0.7, 1.0, 'Sequence', ha='right', fontsize=10, color='#FFC107', weight='bold')

        # delta box (4 delta nodes on right, indices 8-11)
        delta_x_min, delta_x_max = 7.5, 11.5
        delta_y_min, delta_y_max = 0.6, 1.4
        delta_rect = Rectangle((delta_x_min, delta_y_min), delta_x_max - delta_x_min, delta_y_max - delta_y_min,
                               linewidth=2, edgecolor='#FF8A65', facecolor='none', linestyle='--')
        self.ax.add_patch(delta_rect)
        self.ax.text(delta_x_max + 0.7, 1.0, 'Delta', ha='left', fontsize=10, color='#FF8A65', weight='bold')

        self.ax.set_xlim(-1, 12)
        self.ax.set_ylim(-0.5, 2.5)
        self.ax.axis('off')

        self.ax.set_title(f"Epoch {epoch} Batch {batch}")

        self.fig.tight_layout(pad=0.5)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.fig.savefig(self.output_path, dpi=100, bbox_inches='tight')

        # update MAE plot
        self.plot_mae_graph()

        plt.pause(0.01)
