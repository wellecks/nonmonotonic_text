"""Contains `Tree` and `Node` classes used by the oracle, and functions to transform
model samples (level-order token sequences) to an in-order sequence or a tree. """


class Tree(object):
    def __init__(self, root_node, end_idx):
        self.root = root_node
        self.current = root_node
        self.queue = []
        self.END = end_idx

    def next(self):
        if len(self.queue) == 0:
            return None
        node = self.queue[0]
        self.current = node
        self.queue = self.queue[1:]
        return self.current

    def done(self):
        return self.current.value is not None and len(self.queue) == 0

    def generate(self, action):
        self.current.generate(action)
        if self.current.left is not None:
            self.queue.append(self.current.left)
        if self.current.right is not None:
            self.queue.append(self.current.right)

    def print_dfs(self):
        stack = [(self.root, 0)]
        while len(stack) > 0:
            curr, level = stack[-1]
            stack = stack[:-1]
            prefix = '   '*(level-1) + '+--'*(level > 0)
            print("%s%s%s" % (prefix, str(curr.value), '*' if curr == self.current else ''))
            if curr.right:
                stack.append((curr.right, level+1))
            if curr.left:
                stack.append((curr.left, level+1))

    def to_text(self, exclude_end=True):
        tokens = []
        def _inorder(node):
            if node is not None:
                _inorder(node.left)
                if exclude_end and (not node.value == self.END):
                    tokens.append(node.value)
                _inorder(node.right)
        _inorder(self.root)
        return tokens


class Node(object):
    def __init__(self, valid_actions, parent, end_idx, invalid_behavior='split'):
        self.valid_actions = valid_actions
        self.left = None
        self.right = None
        self.value = None
        self.parent = parent
        self.END = end_idx
        self.invalid_behavior = invalid_behavior

    def generate(self, action):
        if self.value is not None:
            raise ValueError('`generate` has already been called at this Node')

        if action in self.valid_actions:
            self.value = action
            # We don't add children if this is an END node (i.e. the only valid action is END).
            if tuple(self.valid_actions) == (self.END,):
                pass
            else:
                # Make a left-child with valid actions (a_1,...,a_{i-1}), where a_i=action,
                # or (END) when there are no valid actions to the left of a_i.
                # NOTE(wellecks): this chooses the left-most instance if there are multiple instances
                idx = self.valid_actions.index(action)
                left_actions = self.valid_actions[:idx] if idx > 0 else [self.END]
                self.left = Node(left_actions, self, self.END, self.invalid_behavior)

                # Make a left-child with valid actions (a_{i+1},...,a_N), where a_i=action,
                # or (END) when there are no valid actions to the right of a_i.
                right_actions = self.valid_actions[idx+1:] if len(self.valid_actions[idx+1:]) > 0 else [self.END]
                self.right = Node(right_actions, self, self.END, self.invalid_behavior)
        else:
            self._handle_invalid(action, self.invalid_behavior)

    def _handle_invalid(self, action, invalid_behavior):
        """Determine this Node's value and its children's valid actions, when `action` is not valid.
        Given valid actions {a, b, c} and invalid `action` d:
            - 'split': Set `d` as this Node's value, create left child {a, b} and right child {c}
            - NOTE(wellecks): can support other invalid behavior handling here
        """
        # Set the invalid action as this Node's value.
        self.value = action

        # Determine children
        if invalid_behavior == 'split':
            # If this is an end node, make both children end nodes.
            if tuple(self.valid_actions) == (self.END,):
                left_actions = (self.END,)
                right_actions = (self.END,)
            # Otherwise split the valid actions in half, and give one half to each child.
            else:
                idx = len(self.valid_actions)//2
                left_actions = self.valid_actions[:idx+1] if len(self.valid_actions[:idx+1]) > 0 else [self.END]
                right_actions = self.valid_actions[idx+1:] if len(self.valid_actions[idx+1:]) > 0 else [self.END]
            self.left = Node(left_actions, self, self.END, self.invalid_behavior)
            self.right = Node(right_actions, self, self.END, self.invalid_behavior)
        else:
            raise NotImplementedError('other invalid behaviors')

    def __str__(self):
        return "Node(Value: %s\tValid Actions: {%s})" % (str(self.value), ', '.join(map(str, self.valid_actions)))

    def __repr__(self):
        return self.__str__()


# --- Data Structures & functions for model samples
class BinaryNode(object):
    def __init__(self, value, level, index, parent, left, right):
        self.value = value
        self.level = level
        self.index = index
        self.parent = parent
        self.left = left
        self.right = right


def build_tree(level_order_tokens):
    """Build a binary tree represented as BinaryNode's from the level-order `level_order_tokens`.
    The returned root can be passed to functions such as `tree_to_text` or `print_tree`.
    """
    root = BinaryNode(level_order_tokens[0], 0, 0, None, None, None)
    i = 1
    queue = [root]
    while len(queue) > 0:
        node = queue[0]
        queue = queue[1:]
        if i < len(level_order_tokens):
            # Set the left child and add to the queue if it is not an end or padding node, otherwise set to None
            if level_order_tokens[i] != '<end>' and level_order_tokens[i] != '<p>':
                child = BinaryNode(level_order_tokens[i], node.level + 1, i, node.index, None, None)
                node.left = child
                queue.append(child)
            else:
                node.left = None
            i += 1
        if i < len(level_order_tokens):
            # Set the right child and add to the queue if it is not an end or padding node, otherwise set to None
            if level_order_tokens[i] != '<end>' and level_order_tokens[i] != '<p>':
                child = BinaryNode(level_order_tokens[i], node.level + 1, i, node.index, None, None)
                node.right = child
                queue.append(child)
            else:
                node.right = None
            i += 1
    return root


def tree_to_text(root):
    """Convert a tree to a sequence of values (e.g. words) via in-order traversal."""
    nodes = []
    def _inorder(node):
        if node.left and node.left.value is not None:
            _inorder(node.left)
        nodes.append(node)
        if node.right and node.right.value is not None:
            _inorder(node.right)
    _inorder(root)
    text = [node.value for node in nodes]
    return text, nodes


def print_tree(root, show_index=False):
    lines = _build_tree_string(root, show_index)[0]
    return '\n' + '\n'.join((line.rstrip() for line in lines))


def _build_tree_string(root, show_index=False):
    # SOURCE: https://github.com/joowani/binarytree
    if root is None:
        return [], 0, 0, 0

    line1 = []
    line2 = []
    if show_index:
        node_repr = '{}-{}'.format(root.index, root.value)
    else:
        node_repr = str(root.value)

    new_root_width = gap_size = len(node_repr)

    # Get the left and right sub-boxes, their widths, and root repr positions
    l_box, l_box_width, l_root_start, l_root_end = \
        _build_tree_string(root.left, show_index)
    r_box, r_box_width, r_root_start, r_root_end = \
        _build_tree_string(root.right, show_index)

    # Draw the branch connecting the current root node to the left sub-box
    # Pad the line with whitespaces where necessary
    if l_box_width > 0:
        l_root = (l_root_start + l_root_end) // 2 + 1
        line1.append(' ' * (l_root + 1))
        line1.append('_' * (l_box_width - l_root))
        line2.append(' ' * l_root + '/')
        line2.append(' ' * (l_box_width - l_root))
        new_root_start = l_box_width + 1
        gap_size += 1
    else:
        new_root_start = 0

    # Draw the representation of the current root node
    line1.append(node_repr)
    line2.append(' ' * new_root_width)

    # Draw the branch connecting the current root node to the right sub-box
    # Pad the line with whitespaces where necessary
    if r_box_width > 0:
        r_root = (r_root_start + r_root_end) // 2
        line1.append('_' * r_root)
        line1.append(' ' * (r_box_width - r_root + 1))
        line2.append(' ' * r_root + '\\')
        line2.append(' ' * (r_box_width - r_root))
        gap_size += 1
    new_root_end = new_root_start + new_root_width - 1

    # Combine the left and right sub-boxes with the branches drawn above
    gap = ' ' * gap_size
    new_box = [''.join(line1), ''.join(line2)]
    for i in range(max(len(l_box), len(r_box))):
        l_line = l_box[i] if i < len(l_box) else ' ' * l_box_width
        r_line = r_box[i] if i < len(r_box) else ' ' * r_box_width
        new_box.append(l_line + gap + r_line)

    # Return the new box, its width and its root repr positions
    return new_box, len(new_box[0]), new_root_start, new_root_end


# --- testing / demo
def generate_random_valid(tokens):
    print("Random Valid Generation:")
    import random
    root = Node(tokens, None, 'ø')
    tree = Tree(root, 'ø')

    go = True
    while go:
        action = random.sample(tree.current.valid_actions, 1)[0]
        tree.generate(action)
        go = tree.next()
    tree.print_dfs()
    return tree


def generate_random_invalid(tokens):
    print("Random Generation with 20% Invalid Actions:")
    import random
    import string
    import numpy as np
    root = Node(tokens, None, 'ø')
    tree = Tree(root, 'ø')

    go = True
    while go:
        action = random.sample(tree.current.valid_actions, 1)[0]
        # generate an invalid action w.p. beta
        if np.random.binomial(1, p=0.2):
            action = ''.join(random.choices(string.ascii_uppercase, k=5))
        tree.generate(action)
        go = tree.next()
    tree.print_dfs()
    return tree


def generate_left_to_right(tokens):
    print("Left-to-Right Generation")
    root = Node(tokens, None, 'ø')
    tree = Tree(root, 'ø')
    go = True
    while go:
        action = tree.current.valid_actions[0]
        tree.generate(action)
        go = tree.next()
    tree.print_dfs()
    return tree


def test_demo():
    text = ['a', 'b', 'c', 'd', 'e']
    t2 = generate_left_to_right(text)
    print(t2.to_text())
    print()
    t1 = generate_random_valid(text)
    print(t1.to_text())
    print()
    t1 = generate_random_invalid(text)
    print(t1.to_text())


def test_binarynode():
    level_order_tokens = ['a', 'b', 'c', 'd', '<end>', '<end>', 'e']
    root = build_tree(level_order_tokens)
    print(print_tree(root))

    level_order_tokens = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    root = build_tree(level_order_tokens)
    print(print_tree(root))

    level_order_tokens = ['a', '<end>', 'b', '<end>', 'c', '<end>', 'd', '<end>', '<end>', '<p>', '<p>']
    root = build_tree(level_order_tokens)
    print(print_tree(root))

    import string
    letters = string.ascii_lowercase
    root = build_tree(letters)
    print(print_tree(root))

if __name__ == '__main__':
    test_demo()
    test_binarynode()
