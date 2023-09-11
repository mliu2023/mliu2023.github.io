import React from 'react';
import PageNavBar from './PageNavBar';
import './Page.css';
import { topics } from './content/Topics';
import { pages } from './content/Pages';

function Page({ unit, index, title }) {
    let currPage = pages[unit - 1][index - 1];
    let pageList = pages[unit - 1];
    let topicList = topics[unit - 1];
    
    return (
        <div className="page">
            <PageNavBar unit={unit} title={title} currIndex={index} pageList={pageList} topics={topicList}/>
            <div className="lesson">
                {currPage}
            </div>
        </div>
    )
}
export default Page;