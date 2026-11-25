"""Matplotlib scientific plots and bounded Pillow animations of actual operations."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from from_scratch import apply_rope, sinusoidal_encoding
from visual_core import METHODS, PERMUTATION, TOKENS, addition_data, alibi_data, attention, fixture, multifrequency_data, permutation_data, precision_data, projected_geometry


BLUE, GREEN, ORANGE, PURPLE, GRAY = "#2473b9", "#168467", "#d47b22", "#8755ad", "#8290a3"
ANIMATIONS = {
    "permutation": ["permutation_equivariance.gif"],
    "sinusoidal": ["sinusoidal_frequencies.gif"],
    "construction": ["positional_encoding_construction.gif"],
    "embedding-space": ["positional_embedding_trajectory.gif"],
    "addition": ["token_position_vector_addition.gif"],
    "similarity": ["relative_distance_similarity.gif"],
    "absolute-relative": ["absolute_vs_relative_positions.gif"],
    "rope": ["rope_rotation.gif", "rope_relative_shift_invariance.gif"],
    "rope-multifrequency": ["rope_multifrequency_rotation.gif"],
    "alibi": ["alibi_distance_penalty.gif"],
    "extrapolation": ["position_interpolation.gif"],
}
STATICS = {
    "permutation": ["attention_with_without_position.png"],
    "construction": ["positional_encoding_heatmap.png"],
    "similarity": ["positional_similarity_matrix.png"],
    "absolute-relative": ["positional_methods_comparison.png"],
    "alibi": ["alibi_attention_heatmaps.png"],
    "comparison": ["attention_mechanisms_comparison.png"],
    "extrapolation": ["rope_numerical_precision.png"],
}


def prepare(group, config):
    data = {}
    if group == "permutation":
        data = permutation_data(config)
    elif group == "embedding-space":
        data = projected_geometry(config)
    elif group == "addition":
        data = addition_data(config)
    elif group == "rope-multifrequency":
        data = multifrequency_data(config)
    elif group == "extrapolation":
        data = precision_data(config)
    elif group in ("absolute-relative", "comparison"):
        x, weights, table = fixture(config, min(config.seq_len, 64))
        data = {"x": x, "weights": weights, "table": table}
    return data


def heat(ax, matrix, title, lo, hi, *, x="Key / position", y="Query / dimension", cmap=None):
    image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", origin="upper", interpolation="nearest",
                      vmin=lo, vmax=hi, cmap=cmap or ("RdBu_r" if lo < 0 else "Blues"))
    ax.set(title=title, xlabel=x, ylabel=y)
    return image


def arrow(ax, vector, origin, color, label):
    ax.annotate("", xy=origin + vector, xytext=origin,
                arrowprops={"arrowstyle": "->", "lw": 2.7, "color": color})
    ax.plot([], [], color=color, label=label)


def draw(fig, group, fraction, config, data, variant=""):
    fig.clear()
    n, d, base = config.seq_len, config.dim, config.base
    fig.set_facecolor("#f3f6fb")
    if group == "permutation":
        stage = int(fraction >= .5)
        key = "permuted" if stage else "original"
        top = fig.add_subplot(3, 1, 1)
        p = PERMUTATION
        destinations = np.argsort(p)
        for token, label in enumerate(TOKENS):
            location = token * (1-fraction) + destinations[token] * fraction
            top.text(location, .5 + .18*np.sin(np.pi*fraction)*(1 if token == 1 else -1 if token == 4 else 0),
                     f"{label}\nID {token}", ha="center", va="center", color=BLUE if token != 4 else GREEN,
                     bbox={"boxstyle": "round,pad=.3", "fc": "white", "ec": "#b6c4d8"})
        top.set(xlim=(-.7, 5.7), ylim=(0, 1), title="Token transport is schematic; attention tensors below are the discrete original/permuted endpoint")
        top.axis("off")
        for row, method in enumerate(("none", "sinusoidal")):
            item = data[method]
            current = item[key]
            matrices = [current["input"], current["logits"], current["probabilities"],
                        item["aligned_output"] if stage else item["original"]["output"]]
            titles = ["Input representations", "Scaled QK logits", "Probabilities", "Inverse-aligned output"]
            for column, (matrix, title) in enumerate(zip(matrices, titles)):
                ax = fig.add_subplot(3, 4, (row+1)*4+column+1)
                heat(ax, matrix, f"{method}\n{title}", 0 if column == 2 else -4,
                     1 if column == 2 else 4, x="Key slot" if column in (1,2) else "Feature", y="Token / query slot")
        fig.suptitle(f"Unmasked self-attention: same projections, synthetic content\n"
                     f"No-position error={data['none']['error']:.2e} | fixed-position difference={data['sinusoidal']['error']:.4f} | joint-coordinate error={data['joint_error']:.2e}", fontsize=13)
    elif group == "sinusoidal":
        positions = np.arange(n)
        pe = sinusoidal_encoding(positions, d, base)
        position = min(n-1, int(fraction*(n-1)))
        ax = fig.add_subplot(2, 1, 1)
        indices = sorted(set([0, 1, d//2-(d//2)%2, d-2, d-1]))
        for index in indices:
            ax.plot(positions, pe[:, index], label=f"dim {index} ({'sin' if index%2==0 else 'cos'})")
        ax.axvline(position, color=PURPLE, ls="--", label=f"position {position}")
        ax.set(xlabel="Token position", ylabel="Encoding value", ylim=(-1.1,1.1), title="Different frequencies; one selected positional vector")
        ax.legend(ncol=3, fontsize=8)
        ax = fig.add_subplot(2, 1, 2)
        ax.bar(np.arange(d), pe[position], color=[BLUE if i%2==0 else GREEN for i in range(d)])
        ax.set(xlabel="Dimension (even = sin, odd = cos)", ylabel="Encoding value", ylim=(-1.1,1.1),
               title=f"p={position}; first pair=[{pe[position,0]:.4f}, {pe[position,1]:.4f}]; norm={np.linalg.norm(pe[position]):.4f}")
        fig.suptitle(f"Sinusoidal encoding · width={d}, base={base:g} · exact coordinates", fontsize=14)
    elif group == "construction":
        if variant.endswith(".png"):
            image = None
            for index, width in enumerate((8,16,32,64,128)):
                ax = fig.add_subplot(2,3,index+1)
                matrix = sinusoidal_encoding(np.arange(n), width, base).T
                image = heat(ax, matrix, f"Width {width}; same base", -1,1, y="Dimension", x="Token position")
            ax = fig.add_subplot(2,3,6)
            ax.axis("off")
            fig.colorbar(image, ax=ax, label="Encoding value; shared scale [-1,1]", shrink=.9)
            ax.text(.05,.5,"Even dimension: sine\nOdd dimension: cosine\nNo independent normalization", transform=ax.transAxes)
        else:
            pe = sinusoidal_encoding(np.arange(n), d, base)
            count = max(1, int(1+fraction*(n-1)))
            visible = pe.T.copy()
            visible[:, count:] = np.nan
            ax = fig.add_subplot(2,1,1)
            image = heat(ax, visible, f"Constructed positions 0–{count-1}", -1,1, x="Token position", y="Dimension")
            fig.colorbar(image, ax=ax, label="Encoding", pad=.02)
            ax = fig.add_subplot(2,1,2)
            ax.bar(np.arange(d), pe[count-1], color=ORANGE)
            ax.set(ylim=(-1.1,1.1), xlabel="Dimension", ylabel="Selected vector value", title=f"Newest position {count-1}")
        fig.suptitle("Building the positional matrix · fixed diverging scale [-1,+1]", fontsize=14)
    elif group == "embedding-space":
        coords = data["coordinates"][:n]
        count = max(2, int(2+fraction*(n-2)))
        ax = fig.add_subplot(1,2,1,projection="3d")
        ax.plot(*coords[:count].T, color=ORANGE)
        ax.scatter(*coords[:count].T, c=np.arange(count), cmap="viridis", vmin=0,vmax=n-1,s=12)
        ax.set(xlabel="PC1",ylabel="PC2",zlabel="PC3",title=f"Fixed PCA basis; {count}/{n} positions")
        for setter, coordinate in ((ax.set_xlim,0),(ax.set_ylim,1),(ax.set_zlim,2)):
            setter(coords[:,coordinate].min()-.1,coords[:,coordinate].max()+.1)
        ax = fig.add_subplot(1,2,2,projection="3d")
        exact=data["exact"][:n]
        ax.plot(*exact[:count].T, color=PURPLE)
        ax.set(xlabel="Original dim 0",ylabel="Original dim 1",zlabel="Original dim 2",title="Exact selected coordinates; no PCA")
        ax.set_xlim(-1.1,1.1);ax.set_ylim(-1.1,1.1);ax.set_zlim(-1.1,1.1)
        explained=np.sum(data["explained_variance_ratio"])
        fig.suptitle(f"Positional trajectory · PCA explained variance {explained:.1%}\nProjection can distort original-dimensional distances",fontsize=14)
    elif group == "addition":
        content=data["content"][0]
        position=data["position"][3]
        alpha=fraction
        ax=fig.add_subplot(1,2,1)
        x,e=content[:2],position[:2]
        bound=max(np.linalg.norm(x),np.linalg.norm(e),np.linalg.norm(x+e))+1
        arrow(ax,x,np.zeros(2),BLUE,"token content")
        arrow(ax,alpha*e,x,ORANGE,"position from content tip")
        arrow(ax,x+alpha*e,np.zeros(2),PURPLE,"partial / complete sum")
        ax.set(xlim=(-bound,bound),ylim=(-bound,bound),xlabel="Exact dimension 0",ylabel="Exact dimension 1",title=f"Vector addition; contribution={alpha:.0%}",aspect="equal")
        ax.grid(alpha=.2);ax.legend(fontsize=8,loc="upper left")
        ax=fig.add_subplot(1,2,2,projection="3d")
        for value,label,c in ((data["projected_content"][0],"same token content",BLUE),
                              (data["projected_combined"][0],"same token at p=0",GREEN),
                              ((content+alpha*position)@data["basis"],"same token + position 3",PURPLE)):
            ax.quiver(0,0,0,*value,color=c,label=label,arrow_length_ratio=.1)
        bound=np.max(np.abs(data["projected_combined"]))+1
        ax.set(xlim=(-bound,bound),ylim=(-bound,bound),zlim=(-bound,bound),xlabel="Linear basis 1",ylabel="Linear basis 2",zlabel="Linear basis 3",title="Shared PCA-derived linear projection; no centering")
        ax.legend(fontsize=8)
        fig.suptitle(f"Content + position preserves width {d}\nOriginal norms: content={np.linalg.norm(content):.3f}; position={np.linalg.norm(position):.3f}; sum={np.linalg.norm(content+position):.3f}",fontsize=14)
    elif group == "similarity":
        pe=sinusoidal_encoding(np.arange(n),d,base)
        matrix=pe@pe.T/(d/2)
        p=min(n-1,int(fraction*(n-1)))
        ax=fig.add_subplot(1,2,1)
        image=heat(ax,matrix,"Pure sinusoidal cosine",-1,1,x="Position r",y="Reference p")
        ax.axhline(p,color=ORANGE,lw=2)
        fig.colorbar(image,ax=ax,shrink=.7)
        ax=fig.add_subplot(1,2,2)
        ax.plot(np.arange(n)-p,matrix[p],color=PURPLE)
        ax.set(xlim=(-(n-1),n-1),ylim=(-1,1),xlabel="Signed displacement r - p (tokens)",ylabel="Cosine similarity",title=f"Moving reference position {p}")
        ax.grid(alpha=.2)
        fig.suptitle("Equal displacement → equal pure positional similarity\nOscillatory geometry does not establish semantic similarity or trained attention",fontsize=14)
    elif group == "absolute-relative":
        shift=int(20*fraction)
        original=np.arange(5,11);moved=original+shift
        a=sinusoidal_encoding(original,d,base);b=sinusoidal_encoding(moved,d,base)
        table=data["table"]
        matrices=[a.T,b.T,table[original].T,table[moved].T,
                  -.08*np.abs(original[:,None]-original[None,:]),-.08*np.abs(moved[:,None]-moved[None,:])]
        titles=["Sinusoidal p=5–10",f"Sinusoidal shifted +{shift}","Random absolute table; original","Random absolute table; shifted","Relative bias; original","Relative bias; shifted (unchanged)"]
        for i,(matrix,title) in enumerate(zip(matrices,titles)):
            ax=fig.add_subplot(2,3,i+1)
            heat(ax,matrix,title,-2 if i in (2,3) else -1,2 if i in (2,3) else 1,x="Pattern slot / key",y="Dimension / query")
        fig.suptitle("Same six-token pattern at different absolute positions\nRandom absolute table is not trained; relative bias depends only on displacement",fontsize=14)
    elif group == "rope":
        common="shift" in variant
        p=2+24*fraction if common else 2
        k=p+3 if common else 2+12*fraction
        w=.5
        from visual_core import rope_planes
        result=rope_planes(p,k,w)
        ax=fig.add_subplot(1,2,1)
        theta=np.linspace(0,2*np.pi,129)
        for radius in (1.,.8):
            ax.plot(radius*np.cos(theta),radius*np.sin(theta),color="#c9d2df",lw=1)
        for key,c,label in (("q",GRAY,"original Q"),("k","#a5b8ac","original K"),("rotated_q",BLUE,"rotated Q"),("rotated_k",GREEN,"rotated K")):
            arrow(ax,result[key],np.zeros(2),c,label)
        for angle,radius,c in ((p*w,.25,ORANGE),((k-p)*w,.45,PURPLE)):
            arc=np.linspace(0,angle,100)
            ax.plot(radius*np.cos(arc),radius*np.sin(arc),color=c,lw=2)
        ax.set(xlim=(-1.4,1.4),ylim=(-1.4,1.4),aspect="equal",xlabel="Pair coordinate 0",ylabel="Pair coordinate 1",title="Exact 2D Q/K rotations; fixed original content")
        ax.legend(fontsize=8,loc="lower left")
        ax=fig.add_subplot(1,2,2)
        if common:
            shifts=np.linspace(0,24,60)
            scores=[rope_planes(2+s,5+s,w)["dot_after"] for s in shifts]
            ax.plot(shifts,scores,color=PURPLE)
            ax.scatter([24*fraction],[result["dot_after"]],color=ORANGE,s=60)
            ax.set(xlabel="Common position shift",ylabel="Unscaled Q·K",ylim=(-1,1),title="Relative displacement stays 3; dot stays constant")
        else:
            offsets=np.linspace(0,12,100)
            scores=[rope_planes(2,2+delta,w)["dot_after"] for delta in offsets]
            ax.plot(offsets,scores,color=PURPLE)
            ax.scatter([k-p],[result["dot_after"]],color=ORANGE,s=60)
            ax.set(xlabel="Key - query displacement",ylabel="Unscaled Q·K",ylim=(-1,1),title="Independent key motion changes the relative score")
        ax.grid(alpha=.2)
        fig.suptitle(f"RoPE {'common-shift invariance' if common else 'relative rotation'} · p={p:.2f}, k={k:.2f}, ω={w:.2f} rad/token\n"
                     f"angles Q={p*w:.2f}, K={k*w:.2f}, relative={(k-p)*w:.2f} rad | norms 1.000, 0.800 | dot={result['dot_after']:.6f}",fontsize=14)
    elif group == "rope-multifrequency":
        position=int(fraction*48)
        q,k=data["q"],data["k"]
        qr=apply_rope(q[None],[2],base)[0];kr=apply_rope(k[None],[position],base)[0]
        contributions=(qr*kr).reshape(-1,2).sum(-1)
        for pair in range(4):
            ax=fig.add_subplot(2,3,pair+1)
            start=2*pair
            if start>=d:
                ax.axis("off");continue
            qpair,kpair=qr[start:start+2],kr[start:start+2]
            bound=max(np.linalg.norm(qpair),np.linalg.norm(kpair))+.3
            arrow(ax,qpair,np.zeros(2),BLUE,"Q");arrow(ax,kpair,np.zeros(2),GREEN,"K")
            ax.set(xlim=(-bound,bound),ylim=(-bound,bound),aspect="equal",title=f"Pair {pair}: ω={base**(-start/d):.4g}\nK angle={position*base**(-start/d):.3f}; dot={contributions[pair]:.3f}")
        ax=fig.add_subplot(2,3,5)
        ax.bar(np.arange(d//2),contributions,color=PURPLE)
        bound=np.sum(np.abs(q*k))+np.linalg.norm(q)*np.linalg.norm(k)
        ax.set(ylim=(-bound,bound),xlabel="Every dimension pair",ylabel="Unscaled dot contribution",title=f"Sum={np.sum(contributions):.4f}; full dot={qr@kr:.4f}")
        ax=fig.add_subplot(2,3,6,projection="3d")
        coords=data["coordinates"]
        ax.plot(*coords[:position+1].T,color=ORANGE)
        ax.set(xlabel="PC1",ylabel="PC2",zlabel="PC3",title="Rotated Q path; fixed PCA basis")
        for setter,index in ((ax.set_xlim,0),(ax.set_ylim,1),(ax.set_zlim,2)):
            setter(coords[:,index].min()-.2,coords[:,index].max()+.2)
        fig.suptitle(f"RoPE has independent 2D planes, not one 3D rotation · key position {position}\nPCA explained variance={np.sum(data['explained_variance_ratio']):.1%}; all {d//2} pairs contribute to score",fontsize=14)
    elif group == "alibi":
        if variant.endswith(".png"):
            from from_scratch import causal_alibi_bias
            from visual_core import softmax
            length=32
            bias=causal_alibi_bias(np.arange(length),np.arange(length),[.1])[0]
            content=np.zeros((length,length));content[:,2]=12
            combined=content+bias
            probabilities=softmax(combined)
            for index,(matrix,title,lo,hi) in enumerate(((bias,"Causal distance bias; slope 0.1",-4,0),
                                                        (np.where(np.isfinite(bias),content,np.nan),"Content logits; key 2 has +12",0,12),
                                                        (combined,"Combined logits after bias",-4,12),
                                                        (probabilities,"Softmax probabilities",0,1))):
                ax=fig.add_subplot(2,2,index+1)
                image=heat(ax,matrix,title,lo,hi)
                fig.colorbar(image,ax=ax,pad=.02,shrink=.8)
            fig.suptitle("ALiBi: finite past penalties and masked future positions\nSame content scores across queries; gray/blank future cells cannot attend",fontsize=14)
            fig.text(.01,.015,"Synthetic / untrained · Educational computation · Interpretation candidate: author review required",fontsize=9,color="#495c77")
            fig.subplots_adjust(left=.07,right=.96,bottom=.13,top=.83,wspace=.4,hspace=.7)
            return
        slope=.02+.98*fraction
        length,query=32,24
        flat=alibi_data(length,query,slope)
        strong=alibi_data(length,query,slope,12,2)
        for column,(name,record) in enumerate((("Flat content",flat),("Distant key 2 has +12 content logit",strong))):
            ax=fig.add_subplot(2,2,column+1)
            finite=np.where(np.isfinite(record["logits"]),record["logits"],np.nan)
            ax.plot(record["content"],label="content",color=BLUE)
            ax.plot(np.where(np.isfinite(record["bias"]),record["bias"],np.nan),label="distance bias",color=ORANGE)
            ax.plot(finite,label="combined",color=PURPLE)
            ax.axvspan(query+.5,length-.5,color=GRAY,alpha=.2,label="future mask")
            ax.set(xlabel="Key position",ylabel="Logit units",title=name,ylim=(-25,15));ax.legend(fontsize=7)
            ax=fig.add_subplot(2,2,column+3)
            ax.bar(np.arange(length),record["probabilities"],color=GREEN if column else BLUE)
            ax.axvspan(query+.5,length-.5,color=GRAY,alpha=.2)
            ax.set(xlabel="Key position",ylabel="Probability",ylim=(0,1),title=f"Expected past distance={record['expected_distance']:.3f}")
        fig.suptitle(f"Causal ALiBi · slope={slope:.3f}, query={query}\nStronger distance penalty favors recency with fixed content; strong content can compete",fontsize=14)
    elif group == "comparison":
        for index,method in enumerate(METHODS):
            record=attention(data["x"],data["weights"],method=method,table=data["table"],base=base,causal=True)
            ax=fig.add_subplot(2,3,index+1)
            heat(ax,record["probabilities"],method,0,1)
        fig.suptitle("Shared synthetic content, projections and causal mask · six mechanisms\nAdditive methods act before Q/K/V; RoPE after Q/K; biases after scaling. No quality ranking.",fontsize=14)
    elif group == "extrapolation":
        if variant.endswith(".png"):
            rows=data["rows"]
            ax=fig.add_subplot(1,2,1)
            for dtype,c in (("float32",ORANGE),("float64",BLUE)):
                ax.loglog([row["position"]+1 for row in rows],[max(row[dtype+"_error"],1e-18) for row in rows],"o-",label=dtype,color=c)
            ax.set(xlabel="Absolute origin + 1",ylabel="Absolute dot error vs relative float64 reference",title="Fixed Q/K content; relative offset 5")
            ax.legend();ax.grid(alpha=.2)
            ax=fig.add_subplot(1,2,2)
            locations=np.arange(len(rows))
            ax.plot(locations,[row["float32_represented_offset"] for row in rows],"o-",color=ORANGE,label="float32 represented offset")
            ax.axhline(5,color=PURPLE,ls="--",label="intended offset 5")
            ax.set_xticks(locations,[str(row["position"]) for row in rows],rotation=30,ha="right")
            ax.set(xlabel="Absolute origin",ylabel="Represented position difference",title="Integer spacing is lost at large float32 indices")
            ax.legend(fontsize=8)
            fig.suptitle("Actual NumPy phase, trigonometry and dot computations in each dtype\nRepresentability and numerical precision do not establish trained-model generalization",fontsize=14)
        else:
            factor=1+31*fraction
            positions=np.linspace(0,4095,128)
            ax=fig.add_subplot(1,2,1)
            ax.plot(positions,positions,color=GRAY,label="original coordinate")
            ax.plot(positions,positions/factor,color=ORANGE,label="effective p / factor")
            ax.axhline(127,color=PURPLE,ls="--",label="hypothetical trained range end")
            ax.set(xlabel="Original position",ylabel="Effective coordinate",ylim=(0,4200),title=f"Coordinate interpolation factor {factor:.2f}")
            ax.legend(fontsize=8)
            ax=fig.add_subplot(1,2,2)
            ax.plot(positions,np.sin(positions),color=BLUE,alpha=.6,label="original high-frequency component")
            ax.plot(positions,np.sin(positions/factor),color=ORANGE,label="interpolated component")
            ax.axvline(128,color=PURPLE,ls="--")
            ax.set(xlabel="Sampled original position",ylabel="Sine coordinate",ylim=(-1.1,1.1),title="128 samples, not a dense long-context attention matrix")
            ax.legend(fontsize=8)
            fig.suptitle("Hypothetical 128-position training boundary · no trained model\nInterpolation changes coordinates and frequency exposure; it is not a quality guarantee",fontsize=14)
    fig.text(.01,.015,"Synthetic / untrained · Educational computation · Interpretation candidate: author review required",fontsize=9,color="#495c77")
    fig.subplots_adjust(left=.07,right=.96,bottom=.13,top=.83,wspace=.45,hspace=.75)


def save_static(folder, group, name, config, data):
    fig=plt.figure(figsize=(13,8),dpi=config.dpi)
    try:
        draw(fig,group,1.,config,data,name)
        fig.savefig(folder/name,dpi=config.dpi,facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)
    return name


def save_animation(folder, group, name, config, data):
    fig=plt.figure(figsize=(12,7.5),dpi=config.dpi)
    try:
        def update(frame):
            fraction=frame/(config.frames-1)
            draw(fig,group,fraction,config,data,name)
            return []
        animation=FuncAnimation(fig,update,frames=config.frames,interval=1000/config.fps,blit=False,repeat=True)
        animation.save(folder/name,writer=PillowWriter(fps=config.fps),dpi=config.dpi)
        for frame in sorted(set([0,config.frames//2,config.frames-1])):
            update(frame)
            target=folder/"frames"/f"{name[:-4]}-{frame:03d}.png"
            target.parent.mkdir(exist_ok=True)
            fig.savefig(target,dpi=config.dpi)
    finally:
        plt.close(fig)
    return name
