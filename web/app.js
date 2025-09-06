// Gini Tree Explorer - Vanilla JS
let treeData = null;
let selectedNodeId = null;

fetch('../tree_data.json')
  .then(r => r.json())
  .then(data => {
    treeData = data.tree;
  trainLabels = data.train_labels;
  renderTree();
  });

function renderTree() {
  const treeDiv = document.getElementById('tree');
  treeDiv.innerHTML = '';
  // Classic tree layout: assign x/y by leaf order and depth
  const nodes = [];
  const links = [];
  let maxDepth = 0;
  let leafCount = 0;
  // First, count leaves for width
  function countLeaves(node) {
    if (!node.left && !node.right) return 1;
    let count = 0;
    if (node.left) count += countLeaves(node.left);
    if (node.right) count += countLeaves(node.right);
    return count;
  }
  const totalLeaves = countLeaves(treeData);
  // Assign positions
  function layout(node, depth, xRange) {
    maxDepth = Math.max(maxDepth, depth);
    let x;
    if (!node.left && !node.right) {
      // Leaf: place at center of xRange
      x = (xRange[0] + xRange[1]) / 2;
    } else {
      // Internal: recurse and average children
      let leftX = xRange[0], rightX = xRange[1];
      let mid;
      if (node.left && node.right) {
        const leftLeaves = countLeaves(node.left);
        const rightLeaves = countLeaves(node.right);
        const split = leftLeaves / (leftLeaves + rightLeaves);
        const leftRange = [xRange[0], xRange[0] + (xRange[1] - xRange[0]) * split];
        const rightRange = [leftRange[1], xRange[1]];
        const leftPos = layout(node.left, depth + 1, leftRange);
        const rightPos = layout(node.right, depth + 1, rightRange);
        mid = (leftPos + rightPos) / 2;
      } else if (node.left) {
        mid = layout(node.left, depth + 1, xRange);
      } else {
        mid = layout(node.right, depth + 1, xRange);
      }
      x = mid;
    }
    const y = 60 + depth * 100;
    nodes.push({
      id: node.id,
      gini: node.gini,
      gini_calc: node.gini_calc,
      samples: node.samples,
      class_counts: node.class_counts,
      x, y, depth
    });
    if (node.left) {
      links.push({from: node.id, to: node.left.id});
    }
    if (node.right) {
      links.push({from: node.id, to: node.right.id});
    }
    return x;
  }
  // Layout in [0, width]
  const width = Math.max(700, totalLeaves * 90);
  layout(treeData, 0, [40, width - 40]);
  const height = 120 + maxDepth * 100;
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('width', width);
  svg.setAttribute('height', height);
  svg.style.overflow = 'visible';
  svg.style.background = '#fff';

  // Draw links (curved)
  links.forEach(link => {
    const from = nodes.find(n => n.id === link.from);
    const to = nodes.find(n => n.id === link.to);
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    const mx = (from.x + to.x) / 2;
    path.setAttribute('d', `M${from.x},${from.y+24} C${from.x},${(from.y+to.y)/2} ${to.x},${(from.y+to.y)/2} ${to.x},${to.y-24}`);
    path.setAttribute('stroke', '#bbb');
    path.setAttribute('stroke-width', 2);
    path.setAttribute('fill', 'none');
    svg.appendChild(path);
  });

  // Draw nodes
  nodes.forEach(node => {
    const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    group.setAttribute('cursor', 'pointer');
    group.addEventListener('click', e => {
      e.stopPropagation();
      selectNode(node.id);
    });
    // Circle
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', node.x);
    circle.setAttribute('cy', node.y);
    circle.setAttribute('r', 24);
    circle.setAttribute('fill', '#fff');
    circle.setAttribute('stroke', '#888');
    circle.setAttribute('stroke-width', 2);
    if (selectedNodeId === node.id) {
      circle.classList.add('selected-node');
    }
    group.appendChild(circle);
    // Gini text
    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', node.x);
    text.setAttribute('y', node.y + 6);
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('font-size', '13px');
    text.setAttribute('fill', '#333');
    text.textContent = node.gini.toFixed(2);
    group.appendChild(text);
    svg.appendChild(group);
  });

  treeDiv.appendChild(svg);
  // Deselect on background click
  svg.addEventListener('click', () => selectNode(null));
}

function selectNode(nodeId) {
  selectedNodeId = nodeId;
  // Highlight selected node
  renderTree();
  // Show info
  const infoDiv = document.getElementById('node-info');
  const gridDiv = document.getElementById('sample-grid');
  if (nodeId === null) {
    infoDiv.textContent = 'Click a node to see details.';
    gridDiv.innerHTML = '';
    return;
  }
  // Find node in tree
  function findNode(node, id) {
    if (node.id === id) return node;
    if (node.left) {
      const l = findNode(node.left, id);
      if (l) return l;
    }
    if (node.right) {
      const r = findNode(node.right, id);
      if (r) return r;
    }
    return null;
  }
  const node = findNode(treeData, nodeId);
  infoDiv.innerHTML = `<b>Node ID:</b> ${node.id}<br>
    <b>Gini:</b> ${node.gini.toFixed(3)}<br>
    <b>Gini Calculation:</b> <br><code>${node.gini_calc}</code><br>
    <b>Class counts:</b> [${node.class_counts.join(', ')}]<br>
    <b>Samples:</b> ${node.samples.length}`;
  // Show sample grid
  gridDiv.innerHTML = node.samples.map(idx => {
    // Use true class label from trainLabels
    const label = trainLabels ? trainLabels[idx] : 0;
    return `<span class="sample-point sample-class-${label}" title="Sample ${idx} (class ${label})"></span>`;
  }).join('');
}

// Helper to get class label for a sample index
function getSampleLabel(idx) {
  // Iris: 0,1,2 in order
  if (idx < 50) return 0;
  if (idx < 100) return 1;
  return 2;
}
