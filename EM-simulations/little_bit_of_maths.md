**1. Gradient (∇𝑓)**

Input: scalar field 
𝑓(𝑥,𝑦,𝑧)

Output: vector field.

Meaning: slope → direction and rate of steepest increase.


∇f=(∂f/∂x, ∂f/∂y, ∂f/∂z)


**2. Divergence (∇⋅𝐹)**
Input: vector field 𝐹 =(𝐹𝑥,𝐹𝑦,𝐹𝑧)
Output: scalar field.

Meaning: how much the field is a source or sink at a point.

Formula:
∇⋅𝐹 =∂Fx/∂x +∂Fy/∂y + ∂Fz/∂z


**3. Curl (∇×𝐹)**
- **Input:** vector field  
  **F** = (Fₓ, Fᵧ, F𝓏)

- **Output:** vector field  

- **Meaning:** tendency of the field to *rotate / swirl* around a point  

- **Formula:**  

  ∇ × **F** = (  
  ∂F𝓏/∂y − ∂Fᵧ/∂z ,  
  ∂Fₓ/∂z − ∂F𝓏/∂x ,  
  ∂Fᵧ/∂x − ∂Fₓ/∂y )  

---

💡 **Intuition:**  
If you place a tiny paddlewheel into the field, the **curl** tells you whether it spins, and if so, around which axis and how strongly.
