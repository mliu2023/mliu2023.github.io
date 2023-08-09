import React from 'react';
import { Route, Routes } from 'react-router-dom';
import { topics } from './Topics';
import Page from './Page';

function SubUnitRoutes() {
    const subUnitRoutes =
        topics.map((list, unitIndex) => (
            <Routes>
                {list.map((title, subUnitIndex) => (
                    <Route path={`/unit${unitIndex + 1}/${subUnitIndex + 1}`}
                        element={<Page unit={unitIndex + 1} index={subUnitIndex + 1} title={title} />}>
                    </Route>))}
            </Routes>));
    return (
        {subUnitRoutes}
    );
}

export default SubUnitRoutes;