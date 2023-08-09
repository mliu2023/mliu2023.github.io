import React from 'react';
import { Route } from 'react-router-dom';
import { topics } from './Topics';
import CustomLink from './CustomLink'

export const subUnitList =
    topics.map((list, unitIndex) => (
        <Route path={`/unit${unitIndex + 1}`}
            element={<Unit unit={unitIndex + 1}
                subUnitList={
                    list.map((title, subUnitIndex) => (
                        <li><CustomLink to={`/unit${unitIndex + 1}/${subUnitIndex + 1}`} className='subunit'>{title}</CustomLink></li>))} />}>
        </Route>));