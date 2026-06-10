import networkx as nx
import matplotlib.pyplot as plt


class NeuralVisualizer:

    def __init__(self, output_path="training_plot.png", smoothing_alpha=0.2, batches_per_epoch=1793):

        self.output_path = output_path
        self.G = nx.DiGraph()

        # store batches per epoch for global batch calculation
        self.batches_per_epoch = batches_per_epoch

        # nodes
        self.seq_input_nodes = [f"S{i+1}" for i in range(4)]
        self.delta_input_nodes = [f"D{i+1}" for i in range(4)]
        self.gru_nodes = [f"G{i}" for i in range(8)]
        self.output_nodes = ["Output"]

        for node in (
            self.seq_input_nodes
            + self.delta_input_nodes
            + self.gru_nodes
            + self.output_nodes
        ):
            self.G.add_node(node)

        # edges: sequence + delta inputs -> hidden GRU layer -> output
        for inp in self.seq_input_nodes + self.delta_input_nodes:
            for gru in self.gru_nodes:
                self.G.add_edge(inp, gru)

        for gru in self.gru_nodes:
            self.G.add_edge(gru, "Output")

        # positions: top row for inputs, middle row for hidden layer
        self.pos = {}
        top_y = 2.2
        for i, n in enumerate(self.seq_input_nodes):
            self.pos[n] = (i * 1.0, top_y)
        for i, n in enumerate(self.delta_input_nodes):
            self.pos[n] = ((len(self.seq_input_nodes) + i) * 1.0, top_y)

        middle_y = 1.0
        for i, n in enumerate(self.gru_nodes):
            self.pos[n] = (i * 1.0, middle_y)

        self.pos["Output"] = ((len(self.gru_nodes) - 1) * 1.0 / 2.0, 0.0)

        # smoothing (EMA)
        self.smoothing_alpha = smoothing_alpha
        self.prev_gru = None
        self.prev_delta = None

        # track MAE history for plotting
        self.mae_history = []
        self.batch_history = []

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
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


    def update(self, epoch, batch, mae, seq_examples, delta_values, gru_values):
        self.ax.clear()

        # track MAE history (global batch counter)
        global_batch = (epoch - 1) * self.batches_per_epoch + batch
        self.mae_history.append(mae)
        self.batch_history.append(global_batch)

        # get magnitudes and apply EMA smoothing
        gru_strength = abs(gru_values[: len(self.gru_nodes)])
        delta_strength = abs(delta_values[: len(self.delta_input_nodes)])

        if self.prev_gru is None:
            self.prev_gru = gru_strength.copy()
        else:
            self.prev_gru = self.smoothing_alpha * gru_strength + (1 - self.smoothing_alpha) * self.prev_gru

        if self.prev_delta is None:
            self.prev_delta = delta_strength.copy()
        else:
            self.prev_delta = self.smoothing_alpha * delta_strength + (1 - self.smoothing_alpha) * self.prev_delta

        gru_strength = self.prev_gru
        delta_strength = self.prev_delta

        # normalization
        max_val = 1e-6
        if len(gru_strength) > 0:
            max_val = max(max_val, float(gru_strength.max()))
        if len(delta_strength) > 0:
            max_val = max(max_val, float(delta_strength.max()))
        for seq_item in seq_examples:
            max_val = max(max_val, abs(float(seq_item[0])), abs(float(seq_item[1])))

        # helpers for color blending
        def hex_to_rgb(h):
            h = h.lstrip('#')
            return tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))

        def rgb_to_hex(rgb):
            return '#%02x%02x%02x' % tuple(int(max(0, min(255, round(c*255)))) for c in rgb)

        def blend_with_white(base_hex, factor):
            br, bg, bb = hex_to_rgb(base_hex)
            wr, wg, wb = (1.0, 1.0, 1.0)
            r = wr * (1-factor) + br * factor
            g = wg * (1-factor) + bg * factor
            b = wb * (1-factor) + bb * factor
            return rgb_to_hex((r, g, b))

        base_seq = '#E3F2FD'   # light blue
        base_delta = '#FFD3A5' # pastel orange
        base_gru = '#FFF59D'   # pastel yellow

        node_colors = []
        labels = {}

        for node in self.G.nodes():
            if node in self.seq_input_nodes:
                idx = self.seq_input_nodes.index(node)
                seq_value = seq_examples[idx]
                labels[node] = f"[{seq_value[0]:.2f},{seq_value[1]:.2f}]"
                factor = min(1.0, (abs(float(seq_value[0])) + abs(float(seq_value[1]))) / (2 * max_val)) * 0.4
                node_colors.append(blend_with_white(base_seq, factor))

            elif node in self.delta_input_nodes:
                idx = self.delta_input_nodes.index(node)
                if idx < len(delta_strength):
                    val = float(delta_strength[idx])
                    labels[node] = f"{val:.2f}"
                    factor = min(1.0, abs(val) / max_val) * 0.5
                    node_colors.append(blend_with_white(base_delta, factor))
                else:
                    labels[node] = '0.00'
                    node_colors.append(blend_with_white(base_delta, 0.1))

            elif node.startswith('G'):
                idx = int(node[1:])
                if idx < len(gru_strength):
                    val = float(gru_strength[idx])
                    labels[node] = f"{val:.3f}"
                    factor = min(1.0, abs(val) / max_val) * 0.6
                    node_colors.append(blend_with_white(base_gru, factor))
                else:
                    labels[node] = '0.000'
                    node_colors.append(blend_with_white(base_gru, 0.0))

            else:
                continue

        visible_nodes = [n for n in self.G.nodes() if n != 'Output']
        visible_edges = [(u, v) for u, v in self.G.edges()]

        nx.draw_networkx_nodes(
            self.G,
            self.pos,
            nodelist=visible_nodes,
            node_color=node_colors,
            node_size=900,
            ax=self.ax
        )

        nx.draw_networkx_labels(self.G, self.pos, labels, ax=self.ax, font_size=8)

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

        # Draw boxes around the sequence and delta input groups
        from matplotlib.patches import Rectangle

        seq_x_min, seq_x_max = -0.5, 3.5
        seq_y_min, seq_y_max = 1.9, 2.4
        seq_rect = Rectangle((seq_x_min, seq_y_min), seq_x_max - seq_x_min, seq_y_max - seq_y_min,
                             linewidth=2, edgecolor='#FFC107', facecolor='none', linestyle='--')
        self.ax.add_patch(seq_rect)
        self.ax.text(seq_x_min - 0.1, (seq_y_min + seq_y_max) / 2, 'Sequence', ha='right', va='center', fontsize=10, color='#FFC107', weight='bold')

        delta_x_min, delta_x_max = 3.5, 7.5
        delta_y_min, delta_y_max = 1.9, 2.4
        delta_rect = Rectangle((delta_x_min, delta_y_min), delta_x_max - delta_x_min, delta_y_max - delta_y_min,
                               linewidth=2, edgecolor='#FF8A65', facecolor='none', linestyle='--')
        self.ax.add_patch(delta_rect)
        self.ax.text(delta_x_min + 4.1, (delta_y_min + delta_y_max) / 2, 'Delta', ha='left', va='center', fontsize=10, color='#FF8A65', weight='bold')

        # Draw a shared top label for the inputs layer
        inputs_x_min = seq_x_min
        inputs_x_max = delta_x_max
        inputs_y = seq_y_max + 0.1
        self.ax.plot([inputs_x_min, inputs_x_max], [inputs_y, inputs_y], color='#999999', linewidth=1)
        self.ax.text((inputs_x_min + inputs_x_max) / 2, inputs_y + 0.05, 'Inputs', ha='center', va='bottom', fontsize=11, weight='bold', color='#333333')

        # Draw box around hidden layer
        hidden_x_min, hidden_x_max = -0.5, 7.5
        hidden_y_min, hidden_y_max = 0.8, 1.3
        hidden_rect = Rectangle((hidden_x_min, hidden_y_min), hidden_x_max - hidden_x_min, hidden_y_max - hidden_y_min,
                                linewidth=2, edgecolor='#E57373', facecolor='none', linestyle='--')
        self.ax.add_patch(hidden_rect)
        self.ax.text(hidden_x_min - 0.1, (hidden_y_min + hidden_y_max) / 2, 'Hidden Layer', ha='right', va='center', fontsize=10, color='#E57373', weight='bold', rotation=90)

        self.ax.set_xlim(-0.6, 7.6)
        self.ax.set_ylim(-0.3, 2.8)
        self.ax.axis('off')

        self.ax.set_title(f"Epoch {epoch} Batch {batch}")

        self.fig.tight_layout(pad=0.5)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.fig.savefig(self.output_path, dpi=100, bbox_inches='tight')

        # update MAE plot
        self.plot_mae_graph()

        plt.pause(0.01)
