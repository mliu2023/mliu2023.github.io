import React from 'react';
import './Unit.css';

function Unit({ unit, subUnitList }){
    const unitNames = ['Introduction to Supervised Learning', 
    'Unsupervised and Other Types of Learning', 
    'State-of-the-art Architectures', 
    'Applications']
    const overview = [
    (
        <article>
            Due to the exponential growth of digital information
            in the past few years, developing algorithms to find
            patterns in data has become essential for informed decision-making.
            Machine learning algorithms learn from the training data and adjust based on
            feedback. The simplest type of learning is supervised learning,
            which is when a model trains on a labeled dataset to predict outputs
            given a set of inputs.
        </article>
    ),
    (
        <article>
            Unit 1 covered supervised learning, which requires labeled datasets. 
            While using labeled datasets is the most reliable way to train a model, 
            the most significant limitation on using labeled datasets is that they 
            are expensive to hand-label. As a result, there is much more unlabeled 
            data than labeled data. This section covers learning methods that can be 
            used with limited or no labeled data.
        </article>
    ),
    (
        <article>
            Not every neural network is a simple multilayer perceptron. 
            Different types of networks have been developed to help models 
            learn different kinds of patterns. For example, convolutional neural 
            networks can learn visual features very well, while recurrent neural 
            networks are designed to learn temporal features. Recently, transformer 
            neural networks have achieved state-of-the-art performance across many 
            domains. In this section, we discuss new architectures and the problems 
            they are designed to solve.
        </article>
    ),
    (
        <article>
            AI has tons of applications. This is not a comprehensive list, and more 
            applications will be added in the future.
        </article>
    )]

    return (
        <div className="unit">
            <h2>Unit {unit} - {unitNames[unit-1]}</h2>
            <p>{overview[unit - 1]}</p>
            <ul className="subunitList">{subUnitList}</ul>
        </div>
    );
};
export default Unit;