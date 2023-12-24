import React from 'react';
import './Unit.css';
import { unitNames, overview } from './content/UnitOverview'

function Unit({ unit, subUnitList }){
    return (
        <div className="unit">
            <h2>Unit {unit} - {unitNames[unit - 1]}</h2>
            {overview[unit - 1]}
            <ul className="subunitList">{subUnitList}</ul>
        </div>
    );
};
export default Unit;