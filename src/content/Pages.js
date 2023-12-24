import ReactEmbedGist from 'react-embed-gist';

export const pages = [
    [
        (
            <article>
                Read the following blog post: <a href='https://vitalflux.com/data-variables-types-uses-in-data-science/'>Types of variables</a>.
                <div class="gistContainer">
                    <ReactEmbedGist className="gist" wrapperClass="gistWrapper" titleClass="gistTitle" gist="mliu2023/3bebbdb262f9d04af6a8a36efad3414f"/>
                </div>
            </article>
        ),
        (
            <article>
                Read the following article: <a href='https://www.statology.org/label-encoding-vs-one-hot-encoding/'>Label Encoding vs. One Hot Encoding: What's the Difference? - Statology</a>. <br/>
                For examples of encoding using sklearn, read through the links in the Additional Resources section at the bottom of the page.
            </article>
        ),
        (
            <article>
                Watch the following video: <a href='https://www.youtube.com/watch?v=swCf51Z8QDo&ab_channel=IntuitiveML'>Intuition: Training Set vs. Test Set vs. Validation Set</a>.
                Pay attention to why the validation set exists.
            </article>
        ),
        (
            <article>
                Watch the following video: <a href='https://www.youtube.com/watch?v=o3DztvnfAJg&ab_channel=NStatum'>Underfitting & Overfitting - Explained</a>.
            </article>
        ),
        (
            <article>
                Watch the following videos for three different ML methods: <br/>
                <a href='https://www.youtube.com/watch?v=cIbj0WuK41w&ab_channel=Econoscent'>Visual Guide to Random Forests</a> <br/>
                <a href='https://www.youtube.com/watch?v=_YPScrckx28&ab_channel=VisuallyExplained'>Support Vector Machine (SVM) in 2 minutes</a> <br/>
                <a href='https://www.youtube.com/watch?v=KluQCQtHTqk&ab_channel=BrandonRohrer'>How k-nearest neighbors works</a>
            </article>
        ),
        (
            <article>
                Watch the full 3Blue1Brown series: <a href='https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&ab_channel=3Blue1Brown'>3b1b series</a>.
                For a more in-depth coverage of neural networks, watch <a href='https://youtu.be/QDX-1M5Nj7s?t=693'>this MIT lecture</a> afterwards.
                <div class="gistContainer">
                    <ReactEmbedGist className="gist" wrapperClass="gistWrapper" titleClass="gistTitle" gist="mliu2023/6960624ec11a2cc5de1bad297cec5e46"/>
                </div>
            </article>
        )
    ],
    [
        (
            <article>
                Watch the following video: <a href='https://www.youtube.com/watch?v=R2e3Ls9H_fc&ab_channel=TheDataPost'>K-means visualized</a>. <br/>
                <a href='https://www.naftaliharris.com/blog/visualizing-k-means-clustering/'>Here</a> is the demo used in the video.
            </article>

        ),
        (

            <article>
                Watch the following video: <a href='https://www.youtube.com/watch?v=FD4DeN81ODY&ab_channel=VisuallyExplained'>Principal Component Analysis (PCA)</a>. <br/>
            </article>

        ),
        (
            <article>
                Watch the following video: <a href='https://www.youtube.com/watch?v=9zKuYvjFFS8&ab_channel=ArxivInsights'>Variational Autoencoders</a> <br/>
                For a higher-level explanation of autoencoders, see <a href='https://www.youtube.com/watch?v=3jmcHZq3A5s&t=457s&ab_channel=WelcomeAIOverlords'>this video</a>. <br/>
                <div class="gistContainer">
                    <ReactEmbedGist className="gist" wrapperClass="gistWrapper" titleClass="gistTitle" gist="mliu2023/a32ba3d43345f43f75044c2cb036efdd"/>
                </div>
            </article>
        ),
        (
            <article>
                Read the following: <a href='https://cs231n.github.io/transfer-learning/'>Transfer learning - CS231n</a>.
                <div class="gistContainer">
                    <ReactEmbedGist className="gist" wrapperClass="gistWrapper" titleClass="gistTitle" gist="mliu2023/9467f4bd11ce6a9016aefb3438f2d4b8"/>
                </div>
            </article>
        ),
        (
            <article>
                Read the following article: <a href='https://towardsdatascience.com/reinforcement-learning-101-e24b50e1d292'>Reinforcement Learning 101</a>. <br/>
                If you would like a more hands-on learning experience, check out this <a href='https://huggingface.co/learn/deep-rl-course/unit0/introduction'>RL course from HuggingFace</a>. <br/>
                <a href='https://www.youtube.com/@b2stud'>b2studios</a> and <a href='https://www.youtube.com/@CodeBullet'>Code Bullet</a> make some entertaining RL simulations.
            </article>
        ),
        (
            <article>
                Read the following article: <a href='https://towardsdatascience.com/multi-task-learning-in-machine-learning-20a37c796c9c'>Multi-task learning</a>.
            </article>
        )
    ],
    [
        (
            <article>
                Watch the following Stanford lectures: <a href='https://www.youtube.com/watch?v=bNb2fEVKeEo&t=3646s&ab_channel=StanfordUniversitySchoolofEngineering'>Lecture 5 | Convolutional Neural Networks</a>, <a href='https://www.youtube.com/watch?v=DAOcjicFr1Y&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=9&ab_channel=StanfordUniversitySchoolofEngineering'>Lecture 9 | CNN Architectures</a>.
            </article>
        ),
        (
            <article>
                Watch the following Stanford lecture: <a href='https://www.youtube.com/watch?v=nDPWywWRIRo&list=RDCMUCdKG2JnvPu6mY1NDXYFfN0g&index=6&ab_channel=StanfordUniversitySchoolofEngineering'>Lecture 11 | Detection and Segmentation</a>. <br/>
                Then, watch the following video from 6:24 to 12:26: <a href='https://youtu.be/sFztPP9qPRc?t=384'>U-Nets Explained</a>.
            </article>
        ),
        (
            <article>
                Watch the following videos: <a href='https://www.youtube.com/watch?v=LHXXI4-IEns&ab_channel=TheA.I.Hacker-MichaelPhi'>RNNs</a>, <a href='https://www.youtube.com/watch?v=8HyCNIVRbSU&ab_channel=TheA.I.Hacker-MichaelPhi'>LSTM and GRU</a>.
                How are LSTMs and GRUs an improvement over standard RNNs?
            </article>
        ),
        (
            <article>
                Watch the following video from 1:09 to 7:40: <a href='https://youtu.be/aw3H-wPuRcw?t=69'>Attention mechanism</a>. <br/>
                Then, watch the following video: <a href='https://youtu.be/TQQlZhbC5ps?t=164'>Transformers</a>.
                Pay attention to how the attention mechanism is different in the encoder and the decoder.
            </article>
        ),
        (
            <article>
                Read the following article: <a href='https://ai.facebook.com/blog/computer-vision-combining-transformers-and-convolutional-neural-networks/'>CNNs vs Transformers</a>.
                Focus on understanding when inductive bias can be beneficial and when it can be detrimental.
            </article>
        ),
        (
            <article>
                Watch the following MIT lecture: <a href='https://youtu.be/3G5hWM6jqPk?t=2335'>Deep Generative Modeling</a>.
            </article>
        ),
        (
            <article>
                Watch the following videos: <a href='https://www.youtube.com/watch?v=HoKDTa5jHvg&ab_channel=Outlier'>Diffusion explained</a>, <a href='https://youtu.be/FHeCmnNe0P8?t=2578'>MIT Diffusion Lecture</a>. <br/>
                If you wanted to see diffusion implemented from scratch, watch <a href='https://www.youtube.com/watch?v=a4Yfz2FxXiY&ab_channel=DeepFindr'>this video</a>.
            </article>
        ),
        (
            <article>
                Read the following document for a high-level overview of graph neural networks: <a href='https://distill.pub/2021/gnn-intro/'>Distill article</a>. <br/>
                Then, watch the following video: <a href='https://www.youtube.com/watch?v=GXhBEj1ZtE8&t=14s&ab_channel=AlexFoo'>Graph Neural Networks</a>.
                Notice the similarity between graph neural networks and convolutional neural networks.
            </article>
        ),
        (
            <article>
                Watch the following MIT lecture: <a href='https://www.youtube.com/watch?v=p1NpGC8K-vs&ab_channel=AlexanderAmini'>The Modern Era of Statistics</a>. <br/>
                In what ways were some of the accepted assumptions from classical statistics inaccurate?
                What improvements have allowed models to use more parameters effectively?
            </article>
        )
    ],
    [
        (
            <article>
                Tesla AI days: <a href='https://www.youtube.com/watch?v=j0z4FweCy4M&ab_channel=Tesla'>2021</a>, <a href='https://www.youtube.com/watch?v=ODSJsviD_SU&ab_channel=Tesla'>2022</a>.
            </article>
        ),
        (
            <article>
                A run-through of computer vision: <a href='https://www.youtube.com/watch?v=sFztPP9qPRc&ab_channel=Gonkee'>Stable Diffusion Explained</a>. 
                The first half of the video explains necessary background from other computer vision tasks.
            </article>
        ),
        (
            <article>
                MIT researchers discover <a href='https://news.mit.edu/2020/artificial-intelligence-identifies-new-antibiotic-0220'>a new antibiotic compound</a>. <br/>
                <a href='https://www.nature.com/articles/s41467-022-29939-5'>Equivariant graph neural networks</a> are used to learn the structure of molecules.
            </article>
        ),
        (
            <article>
                Neural Radiance Fields (NeRFs) are a type of neural network that can synthesize 
                views of complex 3D scenes. <a href='https://www.youtube.com/watch?v=CRlN-cYFxTk&ab_channel=YannicKilcher'>NeRFs Explained</a>.
            </article>
        ),
        (
            <article>
                Adversarial attacks are used to trick a model into producing an 
                incorrect/nonsensical output. <a href='https://www.tensorflow.org/tutorials/generative/adversarial_fgsm'>Tensorflow example</a>. <br/>
                This is particularly useful when we want to prevent large AI models from manipulating our images to create deepfakes. <br/>
                A new tool called <a href='https://www.technologyreview.com/2023/07/26/1076764/this-new-tool-could-protect-your-pictures-from-ai-manipulation/'>PhotoGuard</a> was developed to protect your images. 
                Click <a href='https://github.com/MadryLab/photoguard'>here</a> for the source code.
            </article>
        ),
        (
            <article>
                Large Language Models (LLMs) have been successful at many natural language tasks, 
                but one area in which they have proven to be particularly useful is code generation. <br/>
                Check out <a href='https://github.com/AntonOsika/gpt-engineer'>this GitHub repository</a> for 
                a tool that can generate an entire codebase from one prompt.
            </article>
        ),
        (
            <article>
                AI can be used to extract information from documents. This task is called <a href='https://huggingface.co/tasks/document-question-answering'>Document Visual Question Answering</a>, and 
                corporations are able to use models like <a href='https://medium.com/dair-ai/papers-explained-10-layout-lm-32ec4bad6406'>LayoutLM</a> to automate document processing.
                Check out the following links for some Document AI resources: <br/> <br/>
                <a href='https://github.com/impira/docquery'>Docquery</a> - allows users to ask questions to a single document or directory of files<br/>
                <a href='https://labelstud.io/guide/get_started#Quick-start'>Label Studio</a> - a data labeling tool<br/>
                <a href='https://huggingface.co/blog/document-ai'>Document AI</a> - HuggingFace blog post<br/>
                <a href='https://www.butlerlabs.ai/blog-posts/tutorial-train-layoutlm-on-a-custom-dataset-with-hugging-face'>Fine-tuning on custom dataset</a> - butlerlabs.ai tutorial
            </article>
        )
    ]
];