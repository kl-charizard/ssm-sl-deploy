//
//  Item.swift
//  demo
//
//  Created by Kenny Lam on 12/9/25.
//

import Foundation
import SwiftData

@Model
final class Item {
    var timestamp: Date
    
    init(timestamp: Date) {
        self.timestamp = timestamp
    }
}
