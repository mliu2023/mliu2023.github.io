import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import './PageNavBar.css';

function PageNavBar({ unit, title, currIndex, pageList }) {
    const [pageIndex, setPageIndex] = useState(currIndex);

    // define event handlers 
    const goBack = () => {
        setPageIndex(prevIndex => prevIndex - 1);
    };

    const goNext = () => {
        setPageIndex(prevIndex => prevIndex + 1);
    };

    // determine if on the first question or not 
    const onFirstPage = pageIndex === 1;
    const onLastPage = pageIndex === pageList.length;

    // svgs for arrows
    const leftArrow = <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#969696" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M15 18l-6-6 6-6" /></svg>
    const rightArrow = <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#969696" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 18l6-6-6-6" /></svg>
    
    let backButton = (
        <Link style={{ pointerEvents: onFirstPage ? 'none' : '' }} to={`/unit${unit}/${pageIndex-1}`} className="leftLink">
            <button onClick={goBack} className="leftButton">
                {<>
                    {leftArrow}
                    <div className="buttonText">Prev</div>
                </>}
            </button>
        </Link>
    )
    let nextButton = (
        <Link style={{ pointerEvents: onLastPage ? 'none' : '' }} to={`/unit${unit}/${pageIndex+1}`} className="rightLink">
            <button onClick={goNext} className="rightButton">
                {<>
                    <div className="buttonText">Next</div>
                    {rightArrow}
                </>}
            </button>
        </Link>
    )
    let titleBlock = (
        <div className="title">
            <h3>{title}</h3>
        </div>
    )

    return (
        <div className="pageNav">
            <ul>
                <li>{backButton}</li>
                <li>{titleBlock}</li>
                <li>{nextButton}</li>
            </ul>
        </div>
    );
}
export default PageNavBar;