import React from 'react';
import { Link, useMatch, useResolvedPath } from 'react-router-dom';

function CustomLink({ to, children, ...props }) {
    const resolvedPath = useResolvedPath(to);
    const isActive = useMatch({ path: resolvedPath.pathname, end: false });
    return (
        <li>
            <Link to={to} {...props}>
                {children}
            </Link>
            <div className={isActive ? "active" : ""} />
        </li>
    );
};
export default CustomLink;