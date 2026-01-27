<h1>Adaptive Over-Parameterization</h1>

<p>
Most neural networks are trained under artificial scarcity.
This project explores the opposite: train massively over-parameterized models,
then let the data decide what deserves to survive.
</p>

<p>
If capacity is cheap during training but expensive at inference,
then optimal representations emerge more reliably from excess
than from constraint.
</p>

<h2>What This Pushes Against</h2>
<ul>
  <li>Premature architectural minimalism</li>
  <li>Manual feature selection disguised as “regularization”</li>
  <li>Static model capacity in non-stationary environments</li>
</ul>

<h2>Why This Might Fail</h2>
<ul>
  <li>Over-parameterization may encode spurious correlations</li>
  <li>Pruning signals may lag true feature importance</li>
  <li>Optimization noise could masquerade as usefulness</li>
</ul>

<h3>Core Idea</h3>
<pre>
Over-parameterize →
Expose to data →
Measure utility →
Prune aggressively →
Repeat
</pre>

<blockquote>
Capacity first. Structure later. Prune without mercy.
</blockquote>


