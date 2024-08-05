import React from 'react';
import { Route, Routes } from 'react-router-dom';
import { topics } from '../content/Topics';
import Page from './Page';

export const subUnitRoutes =
    topics.map((list, unitIndex) => (
        <Routes>
            {list.map((title, subUnitIndex) => (
                <Route path={`/unit${unitIndex + 1}/${subUnitIndex + 1}`}
                    element={<Page unit={unitIndex + 1} index={subUnitIndex + 1} title={title} />}>
                </Route>))}
        </Routes>));