# NeRFs-ICCV2023asasjas

- #Papers: 4



-----
[1] Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields
- Category：Anti-Aliased, Grid-Based
- Project: https://jonbarron.info/zipnerf/
- Code: [unofficial](https://github.com/SuLvXiangXin/zipnerf-pytorch)
- Paper: https://arxiv.org/abs/2304.06706
- Abstract:
> *Neural Radiance Field training can be accelerated through the use of grid-based representations in NeRF's learned mapping from spatial coordinates to colors and volumetric density. However, these grid-based approaches lack an explicit understanding of scale and therefore often introduce aliasing, usually in the form of jaggies or missing scene content. Anti-aliasing has previously been addressed by mip-NeRF 360, which reasons about sub-volumes along a cone rather than points along a ray, but this approach is not natively compatible with current grid-based techniques. We show how ideas from rendering and signal processing can be used to construct a technique that combines mip-NeRF 360 and grid-based models such as Instant NGP to yield error rates that are 8% - 77% lower than either prior technique, and that trains 24x faster than mip-NeRF 360.* 
- Figure: 
  ![image-20230716204429598](E:\ResearchFields\NeRFs-CVPR2023\NeRFs-ICCV2023.assets\image-20230716204429598.png)


-----
[2] Delicate Textured Mesh Recovery from NeRF via Adaptive Surface Refinement
- Category：nerf2mesh, nerf-texture
- Project: https://me.kiui.moe/nerf2mesh/
- Code: https://github.com/ashawkey/nerf2mesh
- Paper: https://arxiv.org/pdf/2303.02091.pdf
- Abstract:
> *Neural Radiance Fields (NeRF) have constituted a remarkable breakthrough in image-based 3D reconstruction. However, their implicit volumetric representations differ significantly from the widely-adopted polygonal meshes and lack support from common 3D software and hardware, making their rendering and manipulation inefficient. To overcome this limitation, we present a novel framework that generates textured surface meshes from images. Our approach begins by efficiently initializing the geometry and view-dependency decomposed appearance with a NeRF. Subsequently, a coarse mesh is extracted, and an iterative surface refining algorithm is developed to adaptively adjust both vertex positions and face density based on re-projected rendering errors. We jointly refine the appearance with geometry and bake it into texture images for real-time rendering. Extensive experiments demonstrate that our method achieves superior mesh quality and competitive rendering quality.* 
- Figure: 
  ![image-20230716204836030](E:\ResearchFields\NeRFs-CVPR2023\NeRFs-ICCV2023.assets\image-20230716204836030.png)
  ![image-20230716204902160](E:\ResearchFields\NeRFs-CVPR2023\NeRFs-ICCV2023.assets\image-20230716204902160.png)


-----
[3] IntrinsicNeRF: Learning Intrinsic Neural Radiance Fields for Editable Novel View Synthesis
- Category：Editable
- Project: https://zju3dv.github.io/intrinsic_nerf/
- Code: https://github.com/zju3dv/IntrinsicNeRF
- Paper: https://arxiv.org/pdf/2210.00647.pdf
- Abstract:
> *Existing inverse rendering combined with neural rendering methods~\cite{zhang2021physg, zhang2022modeling} can only perform editable novel view synthesis on object-specific scenes, while we present intrinsic neural radiance fields, dubbed IntrinsicNeRF, which introduce intrinsic decomposition into the NeRF-based~\cite{mildenhall2020nerf} neural rendering method and can extend its application to room-scale scenes. Since intrinsic decomposition is a fundamentally under-constrained inverse problem, we propose a novel distance-aware point sampling and adaptive reflectance iterative clustering optimization method, which enables IntrinsicNeRF with traditional intrinsic decomposition constraints to be trained in an unsupervised manner, resulting in temporally consistent intrinsic decomposition results. To cope with the problem that different adjacent instances of similar reflectance in a scene are incorrectly clustered together, we further propose a hierarchical clustering method with coarse-to-fine optimization to obtain a fast hierarchical indexing representation. It supports compelling real-time augmented applications such as recoloring and illumination variation. Extensive experiments and editing samples on both object-specific/room-scale scenes and synthetic/real-word data demonstrate that we can obtain consistent intrinsic decomposition results and high-fidelity novel view synthesis even for challenging sequences. Project page: [this https URL](https://zju3dv.github.io/intrinsic_nerf).* 
- Figure: 
  ![image-20230716205328607](E:\ResearchFields\NeRFs-CVPR2023\NeRFs-ICCV2023.assets\image-20230716205328607.png)
  ![image-20230716205418023](E:\ResearchFields\NeRFs-CVPR2023\NeRFs-ICCV2023.assets\image-20230716205418023.png)


-----
[4] DreamBooth3D: Subject-Driven Text-to-3D Generation
- Category：Text-to-3D
- Project: https://dreambooth3d.github.io/
- Code: None
- Paper: https://arxiv.org/pdf/2303.13508.pdf
- Abstract:
> *We present DreamBooth3D, an approach to personalize text-to-3D generative models from as few as 3-6 casually captured images of a subject. Our approach combines recent advances in personalizing text-to-image models (DreamBooth) with text-to-3D generation (DreamFusion). We find that naively combining these methods fails to yield satisfactory subject-specific 3D assets due to personalized text-to-image models overfitting to the input viewpoints of the subject. We overcome this through a 3-stage optimization strategy where we jointly leverage the 3D consistency of neural radiance fields together with the personalization capability of text-to-image models. Our method can produce high-quality, subject-specific 3D assets with text-driven modifications such as novel poses, colors and attributes that are not seen in any of the input images of the subject.* 
- Figure: 
  ![image-20230716210037034](E:\ResearchFields\NeRFs-CVPR2023\NeRFs-ICCV2023.assets\image-20230716210037034.png)
  ![image-20230716210104710](E:\ResearchFields\NeRFs-CVPR2023\NeRFs-ICCV2023.assets\image-20230716210104710.png)



-----
[5] 
- Category：
- Project: 
- Code: 
- Paper: 
- Abstract:
> ** 
- Figure: 



-----
[6] 
- Category：
- Project: 
- Code: 
- Paper: 
- Abstract:
> ** 
- Figure: 



-----
[7] 
- Category：
- Project: 
- Code: 
- Paper: 
- Abstract:
> ** 
- Figure: 



-----
[8] 
- Category：
- Project: 
- Code: 
- Paper: 
- Abstract:
> ** 
- Figure: 


-----
[9] 
- Category：
- Project: 
- Code: 
- Paper: 
- Abstract:
> ** 
- Figure: 



-----
[10] 
- Category：
- Project: 
- Code: 
- Paper: 
- Abstract:
> ** 
- Figure: 



-----
[11] 
- Category：
- Project: 
- Code: 
- Paper: 
- Abstract:
> ** 
- Figure: 



-----
[12] 
- Category：
- Project: 
- Code: 
- Paper: 
- Abstract:
> ** 
- Figure: 




-----
[13] 
- Category：
- Project: 
- Code: 
- Paper: 
- Abstract:
> ** 
- Figure: 



-----
[14] 
- Category：
- Project: 
- Code: 
- Paper: 
- Abstract:
> ** 
- Figure: 



-----
[15] 
- Category：
- Project: 
- Code: 
- Paper: 
- Abstract:
> ** 
- Figure: 



-----
[16] 
- Category：
- Project: 
- Code: 
- Paper: 
- Abstract:
> ** 
- Figure: 


-----
[] 
- Category：
- Project: 
- Code: 
- Paper: 
- Abstract:
> ** 
- Figure: 