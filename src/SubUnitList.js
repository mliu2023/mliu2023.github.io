import React from 'react';
import { Route } from 'react-router-dom';
import { topics } from './content/Topics';
import CustomLink from './CustomLink';
import Unit from './Unit';

export const subUnitList =
    topics.map((list, unitIndex) => (
        <Route 
            path={`/unit${unitIndex + 1}`}
            element={
                <Unit unit={unitIndex + 1}
                    subUnitList={
                        list.map((title, subUnitIndex) => (
                        <li>
                            <CustomLink to={`/unit${unitIndex + 1}/${subUnitIndex + 1}`} className='subunit'>
                                {title}
                            </CustomLink>
                        </li>))
                    } 
                />
            }
        />));